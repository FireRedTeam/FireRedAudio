import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config, Qwen3Model
from transformers import initialization as init
from .downsample import Qwen3ClsDownsample

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def make_nonpad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    return ~make_pad_mask(lengths, max_len)

VAE_DOWNSAMPLE_RATE = 960
PATCH_ENCODER_DOWNSAMPLE_RATE = VAE_DOWNSAMPLE_RATE * 4

def pad_to_multiple_of(audio: torch.Tensor):
    target_samples = math.ceil(audio.shape[-1] / PATCH_ENCODER_DOWNSAMPLE_RATE) * PATCH_ENCODER_DOWNSAMPLE_RATE
    pad_len = target_samples - audio.shape[-1]
    if pad_len > 0:
        audio = F.pad(audio, (0, pad_len))
    return audio

def get_vae_and_patch_output_len(audio: torch.Tensor):
    return audio.shape[0] // VAE_DOWNSAMPLE_RATE, audio.shape[0] // PATCH_ENCODER_DOWNSAMPLE_RATE

class RedAEAudioEncoderV1Config(PretrainedConfig):
    model_type = "red_vae_audio_encoder_v1"

    def __init__(
        self,
        # Output
        out_dim: int = 1024,
        # Input reshape
        audio_patch_size: int = 480,    # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
        # Extra downsample
        extra_downsample_rate: int = 2,   # 50Hz -> 25Hz
        downsample_num_hidden_layers: int = 4,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.extra_downsample_rate = extra_downsample_rate
        self.downsample_num_hidden_layers = downsample_num_hidden_layers
        self.initializer_range = initializer_range


class RedAEAudioEncoderV1(PreTrainedModel):
    config_class = RedAEAudioEncoderV1Config
    base_model_prefix = "red_vae_audio_encoder_v1"
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: RedAEAudioEncoderV1Config):
        super().__init__(config)
        self.audio_patch_size = config.audio_patch_size
        self.audio_sample_rate = config.audio_sample_rate
        self.extra_downsample_rate = config.extra_downsample_rate
        self.out_dim = config.out_dim
        self.in_proj = nn.Sequential(
            nn.Linear(self.audio_patch_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.qwen3_config = Qwen3Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            max_window_layers=config.max_window_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            sliding_window=config.sliding_window,
            use_sliding_window=config.use_sliding_window,
            # Follows the outer config so the fallback in loading.py applies.
            _attn_implementation=config._attn_implementation,
            use_cache=False,
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)
        if self.extra_downsample_rate > 1:
            self.downsample = Qwen3ClsDownsample(
                in_dim=config.hidden_size,
                out_dim=config.hidden_size,
                downsample_rate=config.extra_downsample_rate,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.downsample_num_hidden_layers,
                max_position_embeddings=config.max_position_embeddings,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
            )
        self.out_proj = nn.Linear(config.hidden_size, config.out_dim)

        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)

    # --- Encode (inference, pre-padded audio)
    def encode(self, audio: torch.Tensor):
        """
        audio: [B, T_audio] @ sample_rate, pre-padded to a multiple of downsample_rate
        returns: [B, C=out_dim, T_latent]
        """
        audio = pad_to_multiple_of(audio)
        audio_len = torch.full(
            (audio.shape[0],),
            fill_value=audio.shape[1],
            dtype=torch.long,
            device=audio.device,
        )
        with torch.no_grad():
            latents, _ = self.forward(audio, audio_len)  # [B, T, C]
        return latents.transpose(1, 2)  # [B, C, T]

    # --- Forward
    def forward(
        self,
        audio: torch.Tensor,
        audio_len: torch.Tensor,
    ):
        """
        Args:
            audio(torch.Tensor): shape (b, t)
            audio_len(torch.Tensor): shape (b,)
        Returns:

        """
        assert torch.all(audio_len % self.audio_patch_size == 0), \
            'invalid audio_len: {}'.format(audio_len.tolist())
        # Patchify
        xs = audio.unfold(
            dimension=1,
            size=self.audio_patch_size,
            step=self.audio_patch_size,
        )   # (b, num_patch, patch_size) ~ (b, t, c)
        xs_len = audio_len // self.audio_patch_size
        # LLM
        xs = self.in_proj(xs)
        xs_mask = make_nonpad_mask(xs_len, max_len=xs.shape[1])
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=xs_mask,
        )
        xs = outs.last_hidden_state # (b, t, c)
        # Downsample
        if self.extra_downsample_rate > 1:
            xs, xs_len = self.downsample.forward(xs, xs_len)
        xs = self.out_proj(xs)

        return xs, xs_len
