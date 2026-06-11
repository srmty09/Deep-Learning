import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, List, Tuple
from vision_encoder import VisionEncoder


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        lm_model_name: str = "HuggingFaceTB/SmolLM-135M",
        vit_model_name: str = "google/vit-base-patch16-224",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        fusion_type: str = "concat",
        freeze_lm: bool = False,
        num_vision_tokens: int = 64,
    ):
        super().__init__()
        self.num_vision_tokens = num_vision_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(lm_model_name)
        self.lm_config = self.lm.config
        lm_hidden_size = self.lm_config.hidden_size

        self.vision_encoder = VisionEncoder(
            vit_model_name=vit_model_name,
            blip_model_name=blip_model_name,
            lm_hidden_size=lm_hidden_size,
            fusion_type=fusion_type,
        )

        self.vision_token_compressor = nn.Sequential(
            nn.Linear(lm_hidden_size, lm_hidden_size),
            nn.GELU(),
        )

        self.num_vision_queries = num_vision_tokens
        self.vision_query = nn.Parameter(torch.randn(1, num_vision_tokens, lm_hidden_size))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=lm_hidden_size,
            num_heads=8,
            batch_first=True,
        )

        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        special_tokens = {"additional_special_tokens": ["<image>", "</image>", "<|vision_start|>", "<|vision_end|>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.lm.resize_token_embeddings(len(self.tokenizer))

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision_encoder(pixel_values)
        batch_size = pixel_values.shape[0]

        queries = self.vision_query.expand(batch_size, -1, -1)
        compressed, _ = self.cross_attention(queries, vision_features, vision_features)
        compressed = self.vision_token_compressor(compressed)

        return compressed

    def get_lm_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
    ) -> Tuple:
        text_embeddings = self.get_lm_embeddings(input_ids)

        if pixel_values is not None:
            vision_embeddings = self.encode_vision(pixel_values)
            inputs_embeds, attention_mask, labels = self._merge_vision_text(
                text_embeddings, vision_embeddings, input_ids, attention_mask, labels
            )
        else:
            inputs_embeds = text_embeddings

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    def _merge_vision_text(
        self,
        text_embeddings: torch.Tensor,
        vision_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        batch_size, seq_len, hidden_size = text_embeddings.shape
        num_vision_tokens = vision_embeddings.shape[1]

        merged_embeddings = []
        merged_masks = []
        merged_labels = []

        for i in range(batch_size):
            image_token_positions = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(image_token_positions) == 0:
                merged_embeddings.append(text_embeddings[i])
                if attention_mask is not None:
                    merged_masks.append(attention_mask[i])
                if labels is not None:
                    merged_labels.append(labels[i])
                continue

            insert_pos = image_token_positions[0].item()
            before = text_embeddings[i, :insert_pos]
            after = text_embeddings[i, insert_pos + 1:]

            combined = torch.cat([before, vision_embeddings[i], after], dim=0)
            merged_embeddings.append(combined)

            if attention_mask is not None:
                vision_mask = torch.ones(num_vision_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
                merged_mask = torch.cat([attention_mask[i, :insert_pos], vision_mask, attention_mask[i, insert_pos + 1:]])
                merged_masks.append(merged_mask)

            if labels is not None:
                vision_labels = torch.full((num_vision_tokens,), -100, device=labels.device, dtype=labels.dtype)
                merged_label = torch.cat([labels[i, :insert_pos], vision_labels, labels[i, insert_pos + 1:]])
                merged_labels.append(merged_label)

        max_len = max(e.shape[0] for e in merged_embeddings)
        padded_embeddings = torch.zeros(batch_size, max_len, hidden_size, device=text_embeddings.device)
        for i, emb in enumerate(merged_embeddings):
            padded_embeddings[i, :emb.shape[0]] = emb

        final_mask = None
        if merged_masks:
            final_mask = torch.zeros(batch_size, max_len, device=text_embeddings.device, dtype=attention_mask.dtype)
            for i, m in enumerate(merged_masks):
                final_mask[i, :m.shape[0]] = m

        final_labels = None
        if merged_labels:
            final_labels = torch.full((batch_size, max_len), -100, device=text_embeddings.device, dtype=labels.dtype)
            for i, l in enumerate(merged_labels):
                final_labels[i, :l.shape[0]] = l

        return padded_embeddings, final_mask, final_labels

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            prompt = "<image> Describe this image in detail:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                next(self.parameters()).device
            )

        text_embeddings = self.get_lm_embeddings(input_ids)

        if pixel_values is not None:
            vision_embeddings = self.encode_vision(pixel_values)
            inputs_embeds, attention_mask, _ = self._merge_vision_text(
                text_embeddings, vision_embeddings, input_ids, None, None
            )
        else:
            inputs_embeds = text_embeddings
            attention_mask = None

        generated = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        return generated
