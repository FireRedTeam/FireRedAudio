from transformers import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from .audio_encoder.modeling_audio_encoder import (
    FireRedAudioEncoderConfig,
)
from .redae.encoder import RedAEAudioEncoderV1Config
from .flow.estimator import RedDiTConfig
from .flow.patch_encoder import RedPatchEncoderConfig


class FireRedAudioConfig(PretrainedConfig):
    model_type = "firered_audio"

    def __init__(
        self,
        *,
        backbone_config=None,
        audio_encoder_config=None,
        red_vae_config=None,
        patch_encoder_config=None,
        dit_config=None,
        # Special tokens. Defaults match the released tokenizer.json.
        sosp_idx: int = 248077,                                          # <|sosp|>, audio segment start
        eosp_idx: int = 248078,                                          # <|eosp|>, audio segment end
        audio_special_token: str = "<|AUDIO|>",                          # understanding-side placeholder
        audio_special_token_id: int = 248091,
        audio_special_token_no_latent: str = "<|AUDIO_NO_LATENT|>",      # generation-side placeholder
        audio_special_no_latent_id: int = 248092,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_config = self._build(backbone_config, Qwen3_5TextConfig)
        self.audio_encoder_config = self._build(audio_encoder_config, FireRedAudioEncoderConfig)
        self.red_vae_config = self._build(red_vae_config, RedAEAudioEncoderV1Config)
        self.patch_encoder_config = self._build(patch_encoder_config, RedPatchEncoderConfig)
        self.dit_config = self._build(dit_config, RedDiTConfig)

        self.sosp_idx = sosp_idx
        self.eosp_idx = eosp_idx
        self.audio_special_token = audio_special_token
        self.audio_special_token_id = audio_special_token_id
        self.audio_special_token_no_latent = audio_special_token_no_latent
        self.audio_special_no_latent_id = audio_special_no_latent_id

    @staticmethod
    def _build(value, cls):
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls.from_dict(value)
        return value
