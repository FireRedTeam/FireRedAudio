from transformers.models.qwen2_5_omni import Qwen2_5OmniPreTrainedModel
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderLayer,
    SinusoidsPositionEmbedding
)
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers import AutoConfig, AutoModel

from .configuration_audio_encoder import FireRedAudioEncoderConfig

class Adapter(nn.Module):
    def __init__(self, config: FireRedAudioEncoderConfig):
        super().__init__()
        d_model = config.d_model
        output_dim = config.output_dim
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (T, D) -> (D, T) for Conv1d, then back to (T', D) after stride-2 convs
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states

# Adapted from Qwen2.5-Omni's audio encoder (transformers), Apache-2.0
class FireRedAudioEncoder(Qwen2_5OmniPreTrainedModel):
    config: FireRedAudioEncoderConfig

    def __init__(self, config: FireRedAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            [Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.adapter = Adapter(config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _prepare_attention_mask(
        self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        # NOTE: the created attention masl only approximates the ragged FA2 attention by
        # allowing bidirectional attention within `cu_seqlens` blocks, and not attending between
        # blocks. Though it will not be a 100% match for FA2's `varlen` path
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0
        return attention_mask

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        **kwargs: Unpack[TransformersKwargs]
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full(
            (chunk_num.sum(),),
            self.n_window * 2,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = (
            self.padded_and_mask_function(
                chunk_list, chunk_lengths, padding_value=0, padding_side="right"
            )
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states_list = list(hidden_states.split(aftercnn_lens.tolist(), dim=0))

        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.adapter(each_audio_states)
            token_audio_list.append(each_audio_states)

        token_audio = torch.cat(token_audio_list, dim=0)
        return token_audio

    def padded_and_mask_function(
        self, tensor_list, tensor_len, padding_value=0, padding_side="right"
    ):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1  # self.conv2
        output_lengths = (input_lengths - 1) // 2 + 1  # self.adapter.conv3
        output_lengths = (output_lengths - 1) // 2 + 1  # self.adapter.conv4
        return input_lengths, output_lengths


AutoConfig.register("firered_audio_encoder", FireRedAudioEncoderConfig)
AutoModel.register(FireRedAudioEncoderConfig, FireRedAudioEncoder)
