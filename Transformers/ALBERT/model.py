import math
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings_v_to_e = nn.Embedding(config.vocab_size, config.e)
        self.word_embeddings_e_to_h = nn.Linear(config.e, config.d_model, bias=False)

        self.position_embeddings = nn.Embedding(config.seq_len, config.d_model)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.d_model
        )

        self.layernorm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings_v_to_e = self.word_embeddings_v_to_e(input_ids)
        word_embeddings = self.word_embeddings_e_to_h(embeddings_v_to_e)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_ids
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert self.config.d_model % self.config.n_h == 0
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, x, attention_mask):
        batch_size, seq_len, d_model = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(d_model, dim=2)

        q = q.view(
            batch_size, seq_len, self.config.n_h, self.config.d_model // self.config.n_h
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.config.n_h, self.config.d_model // self.config.n_h
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.config.n_h, self.config.d_model // self.config.n_h
        ).transpose(1, 2)

        attention_score = q @ k.transpose(-2, -1) * (1 / math.sqrt(d_model))

        mask = (attention_mask == 0).float()
        mask = mask.unsqueeze(1).unsqueeze(2)
        attention_score = attention_score.masked_fill(mask.bool(), -1e9)
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)

        out = attention_score @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.c_proj(out)
        return out


class ffn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.d_model, config.dff),
            nn.GELU(),
            nn.Linear(config.dff, config.d_model),
            nn.Dropout(config.dp),
        )

    def forward(self, x, attention_mask=None):
        return self.layer(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = CausalSelfAttention(config)
        self.ff = ffn(config)

        self.attn_layernorm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.ff_layernorm = nn.LayerNorm(config.d_model, eps=1e-12)

    def forward(self, x, attention_mask):
        attn_output = self.attn(x, attention_mask)
        x = self.attn_layernorm(x + attn_output)
        ff_output = self.ff(x)
        x = self.ff_layernorm(x + ff_output)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.shared_encoder_block = EncoderBlock(config=config)
        self.n_layers = config.n_layers

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for _ in range(self.n_layers):
            x = self.shared_encoder_block(x, attention_mask)

        return x
