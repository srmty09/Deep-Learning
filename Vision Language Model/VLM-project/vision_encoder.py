import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipProcessor, BlipModel, ViTModel
from typing import Optional, Tuple


class ViTVisionTower(nn.Module):
    def __init__(self, vit_model_name: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_size = self.vit.config.hidden_size
        self.expected_size = self.vit.config.image_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[-1] != self.expected_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.expected_size, self.expected_size),
                mode="bilinear",
                align_corners=False,
            )
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state[:, 1:, :]


class BLIPCaptionEncoder(nn.Module):
    def __init__(self, blip_model_name: str = "Salesforce/blip-image-captioning-base"):
        super().__init__()
        self.blip = BlipModel.from_pretrained(blip_model_name)
        self.expected_size = self.blip.config.vision_config.image_size
        for param in self.blip.parameters():
            param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[-1] != self.expected_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.expected_size, self.expected_size),
                mode="bilinear",
                align_corners=False,
            )
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
        return vision_outputs.last_hidden_state


class VisionEncoder(nn.Module):
    def __init__(
        self,
        vit_model_name: str = "google/vit-base-patch16-224",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        lm_hidden_size: int = 576,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.vit_tower = ViTVisionTower(vit_model_name)
        self.blip_encoder = BLIPCaptionEncoder(blip_model_name)

        vit_dim = self.vit_tower.vit_hidden_size
        blip_dim = self.blip_encoder.blip.config.vision_config.hidden_size

        fused_dim = vit_dim + blip_dim if fusion_type == "concat" else vit_dim

        self.projection = nn.Sequential(
            nn.Linear(fused_dim, lm_hidden_size * 2),
            nn.GELU(),
            nn.Linear(lm_hidden_size * 2, lm_hidden_size),
            nn.LayerNorm(lm_hidden_size),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        proj_dtype = self.projection[0].weight.dtype

        vit_features = self.vit_tower(pixel_values).to(proj_dtype)
        blip_features = self.blip_encoder.encode_image(pixel_values).to(proj_dtype)

        blip_features = blip_features[:, 1:, :]

        min_len = min(vit_features.shape[1], blip_features.shape[1])
        vit_features = vit_features[:, :min_len, :]
        blip_features = blip_features[:, :min_len, :]

        if self.fusion_type == "concat":
            fused = torch.cat([vit_features, blip_features], dim=-1)
        else:
            fused = vit_features + blip_features

        return self.projection(fused)
