import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ViTConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    hidden_dim: int = 512
    n_patches: int = 14*14
    patch_size: int = 16
    n_classes: int = 10
    n_channels: int = 3

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

class PatchEmbedding(nn.Module):
    def __init__(self,config:ViTConfig):
        super().__init__()
        self.cfg = config
        self.conv = nn.Conv2d(
            in_channels=self.cfg.n_channels,
            out_channels=self.cfg.d_model,
            kernel_size=self.cfg.patch_size,
            stride = self.cfg.patch_size
        )
        self.flatten = nn.Flatten(2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.cfg.n_patches + 1, self.cfg.d_model))

    def forward(self,x):
        x = self.conv(x)
        x = self.flatten(x)
        x = x.transpose(-2,-1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self,config:ViTConfig):
        super().__init__()
        self.cfg = config
        self.layernorm1 = nn.LayerNorm(self.cfg.d_model)
        self.layernorm2 = nn.LayerNorm(self.cfg.d_model)
        self.mlp = nn.Sequential(
            *[
                nn.Linear(self.cfg.d_model,self.cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(self.cfg.hidden_dim,self.cfg.d_model)
            ]
        )
    def forward(self,x):
        batch_size,n_patches,d_model = x.shape
        residual1 = x
        x = self.layernorm1(x)
        x = x.reshape(batch_size, n_patches, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            x,
            x,
            x,
            attn_mask=None,
            dropout_p=0.1
        )
        x = x.transpose(1, 2).reshape(batch_size, n_patches, d_model)
        x = x + residual1
        residual2 = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + residual2
        return x

class VisionTransformer(nn.Module):
    def __init__(self,config:ViTConfig):
        super().__init__()
        self.cfg = config
        self.patch_embedding = PatchEmbedding(self.cfg)
        self.encoder = nn.Sequential(
            *[
             VisionTransformerEncoder(self.cfg) for _ in range(self.cfg.n_layers)   
            ]
        )
        self.layernorm = nn.LayerNorm(self.cfg.d_model)
    
    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.layernorm(x)
        return x[:,0]
        
cfg = ViTConfig()
m = VisionTransformer(cfg)
x = torch.randn(2,3,224,224)
print(m(x).shape)