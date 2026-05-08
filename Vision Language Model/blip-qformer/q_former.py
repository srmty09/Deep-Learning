from typing import final
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from dataclasses import dataclass

class DBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def get_embeddings(self, input_ids):
        return self.bert.embeddings(input_ids)

    def forward(self, combined, attention_mask):
        outputs = self.bert.transformer(combined, attn_mask=attention_mask)
        return outputs.last_hidden_state


class CrossAttention(nn.Module):
    def __init__(self, num_queries, q_dim, d_model, num_heads):
        super().__init__()
        self.queries   = nn.Parameter(torch.randn(num_queries, q_dim))
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model
        self.w_q = nn.Linear(q_dim,   d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, q_dim)

    def forward(self, image_emb):
        B  = image_emb.shape[0]
        Np = image_emb.shape[1]
        Nq = self.queries.shape[0]

        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        Q = self.w_q(Q)
        K = self.w_k(image_emb)
        V = self.w_v(image_emb)

        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        score = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn  = torch.softmax(score, dim=-1)
        out   = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        return self.out(out)


@dataclass
class QformerConfig():
    d_model:     int = 768
    q_dim:       int = 256
    num_queries: int = 32
    num_heads:   int = 8


class Qformer(nn.Module):
    def __init__(self, cfg: QformerConfig):
        super().__init__()
        self.cfg    = cfg
        self.ca     = CrossAttention(cfg.num_queries, cfg.q_dim, cfg.d_model, cfg.num_heads)
        self.bert   = DBertModel()
        self.q_proj = nn.Linear(cfg.q_dim, cfg.d_model)

    def forward(self, image_emb, input_ids, attention_mask):
        B       = image_emb.shape[0]
        seq_len = input_ids.shape[1]
        Nq      = self.cfg.num_queries

        visual_queries      = self.ca(image_emb)
        proj_visual_queries = self.q_proj(visual_queries)
        text_emb            = self.bert.get_embeddings(input_ids)
        combined            = torch.cat([proj_visual_queries, text_emb], dim=1)

        q_mask    = torch.ones(B, Nq, Nq + seq_len, device=image_emb.device)
        text_mask = torch.cat([
            torch.zeros(B, seq_len, Nq, device=image_emb.device),
            attention_mask.unsqueeze(1).expand(B, seq_len, seq_len)
        ], dim=2)

        final_attention_mask = torch.cat([q_mask, text_mask], dim=1)

        out = self.bert(combined, final_attention_mask)

        query_out = out[:, :Nq, :]
        text_out  = out[:, Nq:, :]

        return query_out, text_out
