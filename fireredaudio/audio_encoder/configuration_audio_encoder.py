from transformers import PretrainedConfig


class FireRedAudioEncoderConfig(PretrainedConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    n_window (`int`, *optional*, defaults to 100):
        The chunk for conv and flash attn in AudioEncoder.
    output_dim (`int`, *optional*, defaults to 3584):
        The output dimension of AudioEncoder.

    Example:

    ```python
    >>> from fireredaudio.audio_encoder.configuration_audio_encoder import (
    ...     FireRedAudioEncoderConfig)
    >>> from fireredaudio.audio_encoder.modeling_audio_encoder import FireRedAudioEncoder

    >>> config = FireRedAudioEncoderConfig()
    >>> model = FireRedAudioEncoder(config)   # random weights
    ```"""

    model_type = "firered_audio_encoder"
    attribute_map = {"num_hidden_layers": "encoder_layers"}

    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    initializer_range: float = 0.02
    max_source_positions: int = 1500

    n_window: int = 100
    output_dim: int = 3584
