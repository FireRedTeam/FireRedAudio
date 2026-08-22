import torch
import torch.nn.functional as F
from torch import nn
from .modules import (
    TimestepEmbedder,
    Attention,
    FeedForward,
    RMSNorm,
)
from .x_transformers import RotaryEmbedding
from transformers import PretrainedConfig, PreTrainedModel
from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block = torch.nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2
            ),
            nn.Mish(),
            nn.Conv1d(
                out_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=(kernel_size-1)//2
            ),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: shape (b, t, c)
            mask: shape (b, t, 1), default to None
        """
        if mask is not None: x = x * mask
        x = x.transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2)
        if mask is not None: x = x * mask
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


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
        # Attn
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout
        )
        # Conv
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.conv = ConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        # FFN 
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")
        # Time AdaLN condition
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor, rope: torch.Tensor):
        """
        Args:
            x(torch.Tensor): shape (b, t, c)
            c(torch.Tensor): shae (b, 1, c), time condition
            mask(torch.Tensor): default to None, DO NOT USE THIS ARG.
            rope(torch.Tensor): positional embedding.
        """
        # AdaLN for t
        (
            shift_msa, scale_msa, gate_msa, 
            shift_mlp, scale_mlp, gate_mlp, 
            shift_conv, scale_conv, gate_conv
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        # Attn
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            mask=mask, 
            rope=rope
        )
        # Conv
        x = x + gate_conv * self.conv(
            modulate(self.norm2(x), shift_conv, scale_conv), 
            mask=mask,
        )
        # FFN
        x = x + gate_mlp * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


class RedDiTConfig(PretrainedConfig):
    model_type = "red_dit"

    def __init__(
        self,
        # In & Out
        vae_channels: int = 64,
        backbone_hidden_size: int = 4096,
        # Module
        mlp_ratio: float = 4.0,
        depth: int = 11,
        num_heads: int = 16,
        hidden_size: int = 1024,
        # Patch, CFG, ...
        patch_size: int = 4,
        history_patches: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_channels = vae_channels
        self.backbone_hidden_size = backbone_hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = hidden_size 
        self.patch_size = patch_size
        self.history_patches = history_patches


class RedDiT(PreTrainedModel):
    config_class = RedDiTConfig
    base_model_prefix = "red_dit"
    _supports_flash_attn = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config: RedDiTConfig):
        super().__init__(config)
        self.patch_size = config.patch_size
        self.history_patches = config.history_patches
        self.history_length = config.history_patches * config.patch_size
        # Backbone cond proj
        self.backbone_input_proj = nn.Linear(
            config.backbone_hidden_size,
            config.hidden_size,
        )
        self.in_proj = nn.Linear(config.vae_channels+config.hidden_size, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.rotary_embed = RotaryEmbedding(config.hidden_size // config.num_heads)
        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio, dropout=0.1) 
            for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.hidden_size, config.vae_channels)
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            module.reset_parameters()
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, RotaryEmbedding):
            module.rope_init()
            if torch.cuda.is_available():
                module.to(torch.cuda.current_device())
        elif isinstance(module, TimestepEmbedder):
            init.normal_(module.time_mlp[0].weight, std=0.02)
            init.normal_(module.time_mlp[2].weight, std=0.02)
        elif isinstance(module, DiTBlock):
            init.zeros_(module.adaLN_modulation[-1].weight)
            init.zeros_(module.adaLN_modulation[-1].bias)
        elif isinstance(module, FinalLayer):
            init.zeros_(module.adaLN_modulation[-1].weight)
            init.zeros_(module.adaLN_modulation[-1].bias)
            init.zeros_(module.linear.weight)
            init.zeros_(module.linear.bias)

    # --- Inference method
    def _forward_estimator(self, x: torch.Tensor, t: torch.Tensor):
        # time
        t = self.t_embedder(t.view(-1)).unsqueeze(1)  # (b, 1, c)
        x = self.in_proj(x)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, t, None, rope)
        v = self.final_layer(x, t)
        return v

    def generate(
        self,
        backbone_output: torch.Tensor,
        history_vae_latents: torch.Tensor = None,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
    ):
        assert inference_cfg > 0, f'inference_cfg: {inference_cfg} better be>0'
        b = backbone_output.shape[0]

        vae_channels = self.config.vae_channels
        # Make flow timesteps
        t_span = torch.linspace(0, 1, n_timesteps + 1)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi) # (n_timesteps+1,)
        t_span = t_span.to(backbone_output)

        # Prepare history condition
        backbone_pad_len = (self.history_patches+1)-backbone_output.shape[1]
        if backbone_pad_len>0:
            backbone_output = F.pad(backbone_output, (0, 0, backbone_pad_len, 0))
        if history_vae_latents is None:
            history_vae_latents = backbone_output.new_zeros(b, self.history_length, vae_channels)
        elif history_vae_latents.shape[1]<self.history_length:
            history_vae_pad_len = self.history_length-history_vae_latents.shape[1]
            history_vae_latents = F.pad(history_vae_latents, (0, 0, history_vae_pad_len, 0))

        # Compose DiT condition
        dit_backbone_cond = (
            backbone_output[:, -(self.history_patches+1):]
            .repeat_interleave(self.patch_size, dim=1)
        )
        dit_cond = self.backbone_input_proj(dit_backbone_cond)
        # Initial noise
        x0 = torch.cat([
            history_vae_latents[:, -self.history_length:],  # History clean AE latents
            torch.randn(b, self.patch_size, vae_channels, device=history_vae_latents.device, dtype=history_vae_latents.dtype),  # Noise patch
        ], dim=1)
        # CFG
        xt = torch.cat([x0, dit_cond], dim=2)
        xt_cfg = torch.cat([x0, dit_cond * 0], dim=2)
        xt_in = torch.cat([xt, xt_cfg], dim=0)
        
        for tidx, t in enumerate(t_span[:-1]):
            dt = (t_span[tidx+1]-t).expand(b, 1, 1)
            t_in = t.expand(b*2, 1, 1)
            # Estimator
            vt = self._forward_estimator(xt_in, t_in)
            vt = vt[:, self.history_length:]
            # Cfg guidance
            vt_cond, vt_cfg = vt.chunk(2, dim=0)
            vt = (1.0 + inference_cfg) * vt_cond - inference_cfg * vt_cfg
            # Only denoise current patch
            xt_in[:b, -self.patch_size:, :vae_channels] += dt*vt
            xt_in[b:, -self.patch_size:, :vae_channels] = xt_in[:b, -self.patch_size:, :vae_channels]

        one_vae_latents = xt_in[:b, -self.patch_size:, :vae_channels]

        return one_vae_latents
