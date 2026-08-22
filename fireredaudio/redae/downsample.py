import torch
from transformers import Qwen3Config, Qwen3Model


class Qwen3ClsDownsample(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate: int = 2,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 4,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample_rate = downsample_rate
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            _attn_implementation='eager',
            use_cache=False,
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)
        self.cls_tok = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        # Input & output proj
        self.in_proj = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size else
            torch.nn.Identity()
        )
        self.out_proj = (
            torch.nn.Linear(hidden_size, out_dim)
            if out_dim != hidden_size else
            torch.nn.Identity()
        )

    def forward(
        self,
        xs: torch.Tensor,
        xs_len: torch.Tensor,
    ):
        """Downsample latents as BERT [CLS] token

        Args:
            xs(torch.Tensor): shape (b, t, c).
            xs_len(torch.Tensor): shape (b,).
        Returns:
            ys(torch.Tensor): shape (b, t//downsample_rate, c).
            ys_len(torch.Tensor): shape (b,).

        """
        assert torch.all(xs_len % self.downsample_rate == 0), \
            'invalid xs_len: {}'.format(xs_len.tolist())
        b = xs.shape[0]
        # Patchify, (b, t, c) -> (b*t/down, down, c)
        xs = xs.reshape(-1, self.downsample_rate, self.in_dim)
        xs = self.in_proj(xs)
        cls_tok = self.cls_tok.expand(xs.shape[0], -1, -1)
        xs = torch.cat([xs, cls_tok], dim=1)  # (b*t/down, down+1, c)
        # Forward LLM
        outs = self.qwen3(
            inputs_embeds=xs,
        )
        # Take CLS and reshape
        ys = outs.last_hidden_state[:, -1]   # (b*t/down, c)
        ys = self.out_proj(ys)
        ys = ys.reshape(b, -1, self.out_dim)
        ys_len = xs_len // self.downsample_rate

        return ys, ys_len
