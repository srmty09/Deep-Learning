# This file contains the CLIP model implementation

from io import text_encoding
import torch
import torch.nn as nn
from ViT import VisionTransformer,ViTConfig
from TextTransformer import TextTransformer,TextTransformerConfig
from dataclasses import dataclass

@dataclass
class CLIPconfig:
    vit_config: ViTConfig
    text_config: TextTransformerConfig
    projection_dim: int = 1024



class CLIP(nn.Module):
    def __init__(self,config: CLIPconfig):
        super().__init__()
        self.vit = VisionTransformer(config.vit_config)
        self.text = TextTransformer(config.text_config)
        self.image_projection = nn.Linear(config.vit_config.d_model,config.projection_dim)
        self.text_projection = nn.Linear(config.text_config.d_model,config.projection_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self,image,text):
        image_features = self.vit(image)
        text_features = self.text(text)
        image_projected = self.image_projection(image_features)
        text_projected = self.text_projection(text_features)
        logits = torch.matmul(image_projected, text_projected.T) / self.temperature
        return logits


cfg = CLIPconfig(
    vit_config=ViTConfig(),
    text_config=TextTransformerConfig(),
)
model = CLIP(cfg)
print(f"Model parameter count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
x = torch.randn(4, 3, 224, 224)
y = torch.randint(0, 10000, (4, 256))
output = model(x, y)
print(output.shape)
