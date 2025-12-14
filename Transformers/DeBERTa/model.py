import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


@dataclass
class ModelConfig:
    d_model = 512
    n_h = 8
    seq_len = 512
    batch_size = 8
    dff = 2048
    dp = 0.1
    n_layers = 6
    vocab_size = 37000
    k = 2


class DisentagledAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_h == 0
        self.config = config
        self.d_head = config.d_model // config.n_h
        self.Wc = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.Wp = nn.Linear(self.d_head, self.d_head * 2, bias=False)
        self.dropout = nn.Dropout(config.dp)
        self.look_up_table = nn.Parameter(torch.zeros((2 * config.k + 1, self.d_head)))
        nn.init.uniform_(self.look_up_table, -0.1, 0.1)

    def forward(self, x, attention_mask):
        batch_size, seq_len, d_model = x.size()
        d_head = self.d_head
        qkv_c = self.Wc(x)

        qc, kc, vc = qkv_c.split(d_model, dim=2)
        qc = qc.view(batch_size, seq_len, self.config.n_h, d_head).transpose(1, 2)
        kc = kc.view(batch_size, seq_len, self.config.n_h, d_head).transpose(1, 2)
        vc = vc.view(batch_size, seq_len, self.config.n_h, d_head).transpose(1, 2)

        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)
        relative_pos = torch.clamp(j - i, -self.config.k, self.config.k) + self.config.k

        P = self.look_up_table[relative_pos]
        qkv_r = self.Wp(P)
        qr, kr = qkv_r.split(d_head, dim=2)

        c_c = torch.matmul(qc, kc.transpose(-2, -1))
        c_p = torch.einsum("bhid,sid->bhis", qc, kr)
        c_r = torch.einsum("sid,bhjd->bhis", qr, kc)

        attention_score = (c_c + c_p + c_r) / math.sqrt(d_head)
        mask = attention_mask == 0
        mask = mask.unsqueeze(1).unsqueeze(2)

        attention_score = attention_score.masked_fill(mask, -1e9)
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)

        out = torch.matmul(attention_score, vc)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.c_proj(out)
        return out


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.seq_len, config.d_model)
        self.token_type_embeddings = nn.Embedding(config.k, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return BertEmbeddings


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
        self.attention = DisentagledAttention(config)
        self.ffn = ffn(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, x, attention_mask):
        # Self-attention with residual connection
        attn_out = self.attention(x, attention_mask)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList(
            [EncoderBlock(config=config) for _ in range(config.n_layers)]
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size)
        self.nsp_head = nn.Linear(config.d_model, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_output = self.encoder(input_ids, token_type_ids, attention_mask)
        mlm_logits = self.mlm_head(encoder_output)
        cls_output = encoder_output[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)
        return mlm_logits, nsp_logits
