import torch
import torch.nn as nn
from typing import Optional,Tuple,List
import math
from modeling_siglip import SiglipVisionConfig,SiglipVisionModel

class PaliGemmaConfig():
    def __init__(self,
                 vision_config,
                 text_config,
                 ignore_id = -100,
                 image_token_index = 256000,
                 vocab_size = 257152,
                 projection_dim = 2048,
                 pad_token_id = None):
        super().__init__()
        self.ignore_id = ignore_id 
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config,pad_token_id=pad_token_id)

        self.text_config.num_image_tokens = (self.vision_config.image_size//self.vision_config.patch_size)**2
        self.vision_config.projection_dim = projection_dim



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(self,
                input_ids,
                pixel_values,
                attention_mask,
                kv_cache):
        assert torch.all(attention_mask==1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))

        image_features = self.multi_model_projector(image_features)

        input_embeds,attention_mask,position_ids = self._merge_input_ids_with_image_feature(image_features,input_embeds,input_ids,attention_mask,kv_cache)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs
