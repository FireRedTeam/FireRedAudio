"""Model loading.

Two optional accelerations are handled here: flash-attn (backbone and audio encoder)
and liger (backbone RMSNorm / SwiGLU). Both are used when installed and fall back
with a warning otherwise. A fallback changes bf16 reduction order and therefore the
numerical output.
"""

import logging

import torch

from .configuration_fireredaudio import FireRedAudioConfig
from .modeling_fireredaudio import FireRedAudioForCausalLM

logger = logging.getLogger(__name__)


def load_fireredaudio(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> FireRedAudioForCausalLM:
    """Load the model for inference.

    Args:
        model_name_or_path: Directory holding config.json and safetensors shards.
        dtype: Weight dtype; bfloat16 matches the released weights.
        device: Moved there when given.

    Returns:
        A FireRedAudioForCausalLM in eval mode.
    """
    config = FireRedAudioConfig.from_pretrained(model_name_or_path)
    attn = _resolve_attn()

    # dit, patch_encoder and vae/downsample hardcode their attention and ignore this.
    config.backbone_config._attn_implementation = attn
    config.audio_encoder_config._attn_implementation = attn
    config.red_vae_config._attn_implementation = attn

    _apply_liger()

    model = FireRedAudioForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=dtype
    )
    model.eval()
    if device is not None:
        model.to(device)
    return model


def _resolve_attn() -> str:
    """flash_attention_2 when flash-attn is installed, otherwise sdpa."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.warning(
            "flash-attn not installed; falling back to sdpa, which changes the "
            "numerical output."
        )
        return "sdpa"
    return "flash_attention_2"


def _apply_liger() -> bool:
    """Enable liger RMSNorm / SwiGLU if available; returns whether it was applied.

    liger differs from the stock implementation in bf16 reduction order, and that
    difference accumulates over 32 layers. This monkeypatches
    transformers.models.qwen3_5 globally and cannot be undone within a process.
    """
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    except ImportError:
        logger.warning(
            "liger_kernel not installed; RMSNorm/SwiGLU fall back to the stock transformers "
            "implementation, which changes the numerical output."
        )
        return False

    apply_liger_kernel_to_qwen3_5(
        rope=False,
        rms_norm=True,
        swiglu=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    return True
