import math
import random
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore

        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # type: ignore
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) 

        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        
        # flash attention do these 4 lines very fast
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # type: ignore
        # att = F.softmax(att, dim=-1)
        # y = att @ v 

        # flash attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal = True)




        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight # pyright: ignore[reportAttributeAccessIssue]

        self.apply(self._init_weights)

    def _init_weights(self,module):
      if isinstance(module,nn.Linear):
        std = 0.02
        if hasattr(module,"NANOGPT_SCALE_INIT"):
          std *= (2*self.config.n_layer)**-0.5
        torch.nn.init.normal_(module.weight,mean = 0.0, std = std)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
          torch.nn.init.normal_(module.weight,mean = 0.0, std = 0.02)
    @classmethod
    def from_pretrained(cls,model_type):
        "loads pretrained gpt-2 model weights from huggingface"
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
    
        print(f"loading weights from pretrained gpt: {model_type}")
    
        config_args = {
            'gpt2': dict(n_layer=12,n_head=12,n_embd=768),
            'gpt2-medium': dict(n_layer=24,n_head=16,n_embd=1024),
            'gpt2-large': dict(n_layer=36,n_head=20,n_embd=1280),
            'gpt2-xl': dict(n_layer=48,n_head=25,n_embd=1600),
        }[model_type]
    
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
    
        config = GPT2Config(**config_args)
        model = GPT(config=config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
    
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
    
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
    
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} vs {len(sd_keys)}"
    
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
    
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
    
        return model
        
    def forward(self,x,targets=None):
        batch_size,seq_len = x.size()
    
        assert seq_len<=self.config.block_size, f"can't forward"
        pos = torch.arange(0,seq_len,dtype=torch.long,device=x.device)
        pos_emb = self.transformer.wpe(pos) # type: ignore
        tok_emb = self.transformer.wte(x) # pyright: ignore[reportCallIssue]
        x = tok_emb + pos_emb
        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x) # type: ignore
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
          loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss



# num_return_seq = 5
# max_length = 100


# model = GPT.from_pretrained("gpt2")
# model.eval()
# model.to("cuda")

# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("The meaning of life")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
# x = tokens.to("cuda")

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)


# for i in range(num_return_seq):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
