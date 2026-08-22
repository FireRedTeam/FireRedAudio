import torch
import torch.nn as nn
from .modules import (
    Attention,
    FeedForward,
    RMSNorm,
)
from .x_transformers import RotaryEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer


class DiTBlock(GradientCheckpointingLayer):
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0, 
        dropout=0.1, 
        **kwargs
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(self, x, mask, rope):
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class RedPatchEncoderConfig(PretrainedConfig):
    model_type = "red_patch_encoder"

    def __init__(
        self,
        # Input & Output
        vae_dim: int = 64,
        semantic_dim: int = 0,  # NOTE disable semantic input
        out_dim: int = 4096,
        patch_size: int = 4,
        hidden_size: int = 1024,
        mlp_ratio: int = 4,
        depth: int = 8,
        num_heads: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_dim = vae_dim
        self.semantic_dim = semantic_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.num_heads = num_heads


class RedPatchEncoder(PreTrainedModel):
    config_class = RedPatchEncoderConfig
    base_model_prefix = "red_patch_encoder"
    _supports_flash_attn = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config: RedPatchEncoderConfig):
        super().__init__(config)
        self.vae_dim = config.vae_dim
        self.semantic_dim = config.semantic_dim
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.out_dim = config.out_dim
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.num_heads = config.num_heads
        # Input proj
        self.in_proj = torch.nn.Sequential(
            torch.nn.Linear(
                self.vae_dim+self.semantic_dim, 
                self.hidden_size,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
        )
        # [CLS] token
        self.cls_tok = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.rotary_embed = RotaryEmbedding(self.hidden_size // self.num_heads)
        self.blocks = torch.nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio) 
            for _ in range(self.depth)
        ])
        # Output proj
        self.out_proj = FinalLayer(self.hidden_size, self.out_dim)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, RotaryEmbedding):
            module.rope_init()
            if torch.cuda.is_available():
                module.to(torch.cuda.current_device())
        elif isinstance(module, RedPatchEncoder):
            # cls_tok is an nn.Parameter, so it is handled at the root level here.
            init.zeros_(module.cls_tok)

    def forward(
        self,
        inputs_embeds_vae: torch.Tensor,
        inputs_embeds_semantic: torch.Tensor = None,    # NOTE default without semantic feature
    ):
        """Patch encoder aggregating {patch_size} latents into one.

        Args:
            inputs_embeds(torch.Tensor): shape (b, t, c).
        Returns:
            hidden_states(torch.Tensor): shape (b, t//patch_size, c).
        """      
        b = inputs_embeds_vae.shape[0]
        # Proj VAE & Semantic 
        if inputs_embeds_semantic is not None:
            inputs_embeds = torch.cat([
                inputs_embeds_vae, 
                inputs_embeds_semantic,
            ], dim=2)
        else:
            inputs_embeds = inputs_embeds_vae
        inputs_embeds = self.in_proj(inputs_embeds)
        # Patchify, (b, t, c) -> (b*t//patch_size, patch_size, c)
        hidden_states = inputs_embeds.reshape(-1, self.patch_size, self.hidden_size)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)   # (b*t//patch_size, 1, c)
        hidden_states = torch.cat([cls_tok, hidden_states], dim=1)  # (b*t//patch_size, 1+patch_size, c)
        # NOTE full attention
        rope = self.rotary_embed.forward_from_seq_len(hidden_states.shape[1])
        for block in self.blocks:
            hidden_states = block(hidden_states, None, rope)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = hidden_states[:, 0]    # (b*t//patch_size, c)
        hidden_states = hidden_states.reshape(b, -1, self.out_dim)

        return hidden_states