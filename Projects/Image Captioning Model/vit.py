"""
Vision Transformer (ViT) - Image Encoder
Encodes an image into a sequence of patch embeddings via transformer blocks.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class VitConfig:
    patch_size: int = 16
    image_dim: int = 224
    n_head: int = 12
    hidden_dim: int = 768
    layers: int = 12
    mlp_size: int = 3072
    layer_norm_eps: float = 1e-5
    dropout: float = 0.2


class VitPatching(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding=0,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            "Image dimensions must be divisible by patch size"
        )
        # (B, hidden_dim, H', W')
        x = self.conv(img)
        # (B, hidden_dim, N)  →  (B, N, hidden_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class Norm(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm(x)


class MHA(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_head     = cfg.n_head
        self.head_dim   = cfg.hidden_dim // cfg.n_head

        self.qkv      = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        qkv = self.qkv(x)                                           
        q, k, v = torch.split(qkv, self.hidden_dim, dim=2)       

        q = q.view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_head, self.head_dim).transpose(1, 2)

        attn = nn.functional.scaled_dot_product_attention(q, k, v) 
        attn = attn.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)

        return self.out_proj(attn)


class Residual(nn.Module):
    def forward(self, sublayer_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return sublayer_out + residual


class MLP(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.fc1     = nn.Linear(cfg.hidden_dim, cfg.mlp_size)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(cfg.mlp_size, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class VitBlock(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()
        self.norm1    = Norm(cfg)
        self.attn     = MHA(cfg)
        self.res1     = Residual()
        self.norm2    = Norm(cfg)
        self.mlp      = MLP(cfg)
        self.res2     = Residual()

        self.dropout  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(self.dropout(self.attn(self.norm1(x))), x)
        x = self.res2(self.dropout(self.mlp(self.norm2(x))), x)
        return x



class VitEncoder(nn.Module):
    def __init__(self, cfg: VitConfig):
        super().__init__()

        n_patches = (cfg.image_dim // cfg.patch_size) ** 2   
        seq_len   = n_patches + 1      

        self.patching = VitPatching(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, cfg.hidden_dim))
        self.embed_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.Sequential(*[VitBlock(cfg) for _ in range(cfg.layers)])
        self.norm = Norm(cfg)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]

        x = self.patching(img)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        x = self.embed_dropout(x)

        x = self.blocks(x)

        x = self.norm(x)

        return x[:, 0]          
