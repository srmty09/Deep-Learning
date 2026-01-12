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
        self.self_attn = BartMHA(config)
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