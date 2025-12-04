import torch 
import torch.nn as nn 
import torch.nn.functional as F
from dataclasses import dataclass
import math



@dataclass
class ModelConfig:
    d_model = 512 # embeded dim
    n_h = 8 # number of head
    seq_len = 512 # sequence length
    batch_size = 8 # batch size
    dff = 2048 # hidden dim
    dp = 0.1 # dropout
    n_layers = 6 # number of layer
    vocab_size = 37000
    k = 2




class CausalAttention(nn.Module):
    def __init__(self, config, decoder=True):
        super().__init__()
        assert config.d_model % config.n_h == 0
        self.config = config
        self.decoder = decoder
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)
        
        d_k = config.d_model // config.n_h

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len)
        )   
        self.look_up_table_k = nn.Parameter(torch.zeros((2 * config.k + 1, d_k)))
        self.look_up_table_v = nn.Parameter(torch.zeros((2 * config.k + 1, d_k)))

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(d_model, dim=2)
        d_k = k.size(-1) // self.config.n_h
        
        q = q.view(batch_size, seq_len, self.config.n_h, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.n_h, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.n_h, d_k).transpose(1, 2)
        
        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0) 
        
        relative_pos = torch.clamp(j - i, -self.config.k, self.config.k) + self.config.k
        
        aij_k = self.look_up_table_k[relative_pos]
        aij_v = self.look_up_table_v[relative_pos]
        
        raw_attention_score = q @ k.transpose(-2, -1)
        relative_pos_score = torch.einsum('bhid,ijd->bhij', q, aij_k)
        
        attention_score = raw_attention_score + relative_pos_score
        attention_score = attention_score * (1.0 / math.sqrt(d_k))
        
        if self.decoder:
            attention_score = attention_score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf') # type: ignore
            )

        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        
        standard_out = attention_score @ v
        relative_v_out = torch.einsum('bhij,ijd->bhid', attention_score, aij_v)
        
        out = standard_out + relative_v_out
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.c_proj(out)
        return out
    



class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_h == 0
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, q, k, v):
        batch, seq_len, d_model = q.size()

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(batch, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        k = k.view(batch, k.size(1), self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        v = v.view(batch, v.size(1), self.config.n_h, d_model // self.config.n_h).transpose(1, 2)

        attention_score = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        out = attention_score @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.c_proj(out)
        return out


class Residual(nn.Module):
    def __init__(self, sublayer, config):
        super().__init__()
        self.sublayer = sublayer(config)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        return x + self.sublayer(self.ln(x))
    
class ffn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.d_model, config.dff),
            nn.ReLU(),
            nn.Linear(config.dff, config.d_model),
            nn.Dropout(config.dp)
        )

    def forward(self, x):
        return self.layer(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.Sequential(
            Residual(lambda cfg: CausalAttention(cfg, decoder=False), config=config),
            Residual(ffn, config=config)
        )

    def forward(self, x):
        return self.layer(x)
    
class Encoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.config = config
        self.embedding = shared_embedding
        self.layers = nn.Sequential(*[EncoderBlock(config=config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.ln(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mha = Residual(CausalAttention, config)
        self.ca = CrossAttention(config)
        self.feedforward = Residual(ffn, config)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, dec_inp, enc_output):
        y = self.mha(dec_inp)
        out = self.ca(y, enc_output, enc_output)
        out = y + self.ln(out)
        out = self.feedforward(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.config = config
        self.embedding = shared_embedding
        self.layers = nn.ModuleList([DecoderBlock(config=config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, dec_inp, enc_output):
        dec_inp = self.embedding(dec_inp)
        for layer in self.layers:
            dec_inp = layer(dec_inp, enc_output)
        dec_inp = self.ln(dec_inp)
        return dec_inp
    

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Encoder(config, self.shared_embedding)
        self.decoder = Decoder(config, self.shared_embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
    
    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        logits = self.lm_head(dec_out)
        return logits


config = ModelConfig()
model = Model(config)

src = torch.randint(0, config.vocab_size, (4, 512))
tgt = torch.randint(0, config.vocab_size, (4, 512))

output = model(src, tgt)
print("Model output shape:", output.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))