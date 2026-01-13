import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class BartConfig:
    maximum_seq_len: int = 512
    d_model: int = 768
    num_layer: int = 6
    n_head: int = 12 
    vocab_size: int = 30000
    attention_dropout_probs: float = 0.1
    hidden_size: int = 3072

class AddNorm(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.attention_dropout_probs)

    def forward(self, x, residual):
        return self.ln(residual + self.dropout(x))

class BartMHA(nn.Module):
    def __init__(self, config: BartConfig, causal_mask: bool = False):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)
        self.cproj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.causal_mask = causal_mask
        
    def forward(self, tokens, enc_hidden_states=None):
        B, T, C = tokens.shape
        
        if enc_hidden_states is None:  
            q = self.wq(tokens)
            k = self.wk(tokens)
            v = self.wv(tokens)
            T_kv = T
        else:
            k = self.wk(enc_hidden_states)
            v = self.wv(enc_hidden_states)
            q = self.wq(tokens)
            T_kv = enc_hidden_states.shape[1] 
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal_mask)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.cproj(out)
        return out

class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.self_attn = BartMHA(config)
        self.ffn = BartFFN(config)
        self.add_norm1 = AddNorm(config)
        self.add_norm2 = AddNorm(config)
    
    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.add_norm1(attn_out, x)
        
        ffn_out = self.ffn(x)
        x = self.add_norm2(ffn_out, x)
        return x

class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.self_attn = BartMHA(config,causal_mask=True)
        self.cross_attn = BartMHA(config) 
        self.ffn = BartFFN(config)
        self.add_norm1 = AddNorm(config)
        self.add_norm2 = AddNorm(config)
        self.add_norm3 = AddNorm(config)

    def forward(self, x, enc_hidden_states):
        x = self.add_norm1(self.self_attn(x), x)
        
        x = self.add_norm2(self.cross_attn(x, enc_hidden_states), x)
        
        x = self.add_norm3(self.ffn(x), x)
        return x

class BartEncoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.num_layer)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BartDecoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.num_layer)])
    
    def forward(self, x, enc_hidden_states):
        for layer in self.layers:
            x = layer(x, enc_hidden_states)
        return x

class BartModel(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

    def forward(self, encoder_inp, decoder_inp):
        enc_hidden_states = self.encoder(encoder_inp)
        output = self.decoder(decoder_inp, enc_hidden_states)
        return output