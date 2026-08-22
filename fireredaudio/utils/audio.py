"""Audio loading."""

import torch
import torchaudio

# The audio encoder consumes 16 kHz mel; RedAE operates at 24 kHz.
UNDERSTAND_SAMPLE_RATE = 16000
GENERATION_SAMPLE_RATE = 24000


def read_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load as a 1-D mono waveform resampled to `target_sample_rate`."""
    audio, ori_sr = torchaudio.load(path)          # (C, T)
    audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]
    if ori_sr != target_sample_rate:
        audio = torchaudio.functional.resample(audio, ori_sr, target_sample_rate)
    return audio
