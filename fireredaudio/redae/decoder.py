import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import Qwen3Config, Qwen3Model


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


# ISTFT and ISTFTHead below are adapted from Vocos.
# Source: https://github.com/gemelo-ai/vocos
# Licensed under the MIT License. Copyright (c) 2023 Charactr Inc.
class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        win_length: int, 
        padding: str = "same"
    ):
        super().__init__()
        assert padding in ["center", "same"], "Padding must be 'center' or 'same'."
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, 
        dim: int, 
        n_fft: int, 
        hop_length: int, 
        padding: str = "same"
    ):
        super().__init__()
        self.hop_length = hop_length
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor, x_len:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        if x_len is not None:
            audio_length = x_len * self.hop_length
        else:
            audio_length = None
        return audio, audio_length


@dataclass
class PretrainedRedAEDecoderConfig:
    in_dim: int = 64
    upsample_rate: int = 2 # 25Hz -> 50Hz
    # Output reshape
    audio_patch_size: int = 480    # 50Hz
    audio_sample_rate: int = 24000
    # Qwen(mirrors encoder)
    hidden_size: int = 896
    intermediate_size: int = 3584
    num_hidden_layers: int = 18
    max_position_embeddings: int = 32768
    max_window_layers: int = 0
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    sliding_window: int = 64
    use_sliding_window: bool = True


class RedAEAudioDecoderV1(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2, # 25Hz -> 50Hz
        # Output reshape
        audio_patch_size: int = 480,    # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen(mirrors encoder)
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 18,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
    ):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        # Upsample MLP
        self.in_proj = nn.Linear(in_dim, upsample_rate * hidden_size)
        # Qwen3
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)

        self.istft_head = ISTFTHead(
            dim=hidden_size, 
            n_fft=audio_patch_size*4,
            hop_length=audio_patch_size,
            padding='same',
        )

    def forward(
        self,
        xs: torch.Tensor,
        xs_len: torch.Tensor,
    ):
        """
        Args:
            xs(torch.Tensor): shape (b, t, c)
            xs_len(torch.Tensor): shape (b,)
        Returns:
            audio(torch.Tensor): shape (b, t*upsample_rate*patch_size)
            audio_len(torch.Tensor): shape (b,)
        """
        # Upsample
        xs = self.in_proj(xs)   # (b, t, upsample_rate, c)
        xs = xs.reshape(xs.shape[0], -1, self.qwen3_config.hidden_size) # (b, t*upsample_rate, c)
        xs_len = xs_len * self.upsample_rate
        # Qwen3
        xs_mask = make_nonpad_mask(xs_len, max_len=xs.shape[1])
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=xs_mask,
        )
        xs = outs.last_hidden_state # (b, t*upsample_rate, c)

        audio, audio_len = self.istft_head(xs, xs_len)

        return audio, audio_len


class PretrainedRedAEAudioDecoderV1(RedAEAudioDecoderV1):
    @classmethod
    def from_config(cls, config: PretrainedRedAEDecoderConfig)->'PretrainedRedAEAudioDecoderV1':
        decoder = cls(
            in_dim=config.in_dim,
            upsample_rate=config.upsample_rate,
            audio_patch_size=config.audio_patch_size,
            audio_sample_rate=config.audio_sample_rate,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            max_window_layers=config.max_window_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            sliding_window=config.sliding_window,
            use_sliding_window=config.use_sliding_window,
        )
        return decoder

    @classmethod
    def from_pretrained(cls, config: PretrainedRedAEDecoderConfig, ckpt_path:str)->'PretrainedRedAEAudioDecoderV1':
        decoder = cls.from_config(config)
        sd = torch.load(ckpt_path, weights_only=True, map_location='cpu')['model']
        # Skip semantic_proj params
        sd = {
            k.removeprefix('decoder.'): v 
            for k, v in sd.items() 
            if k.startswith('decoder.')
        }
        decoder.load_state_dict(sd, strict=True)
        decoder.eval()
        return decoder

    def decode(self, latents: torch.Tensor):
        latents_len = torch.tensor([latents.shape[1]], dtype=torch.long, device=latents.device)
        with torch.no_grad():
            audio, _ = self.forward(latents, latents_len)
        return audio
