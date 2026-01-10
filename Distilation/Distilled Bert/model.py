import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F


from transformers import BertModel,BertConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

TeacherModel = BertModel.from_pretrained("bert-base-uncased")

bertconfig = BertConfig()

@dataclass
class DistilledBertConfig:
    attention_probs_dropout_prob: float = 0.1
    hidden_size: int = 768
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    type_vocab_size: int = 2
    vocab_size: int = 30522
    device = "cuda" if torch.cuda.is_available() else "cpu"



class BertEmbeddings(nn.Module):
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        # look-up table for the word embedding 
        self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size)

        # look-up table for positon embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)

        # layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)

        # drop-out
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, tokens):
        # token shape: (BatchSize,SeqLen)
        batch_dim, seq_len = tokens.shape

        word_embd = self.word_embeddings(tokens) # (B,S,Embed dim)

        position_ids = torch.arange(0,seq_len,dtype=torch.long,device=self.cfg.device) # (seq_len)
        # added dim to the position dim
        position_ids.unsqueeze(0).expand(batch_dim) # (batch_dim,seq_len)

        pos_embed = self.position_embeddings(position_ids)

        embeddings = word_embd + pos_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    

class BertSelfAttention(nn.Module):
    # Q, K, V projections + attention computation
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, tokens):
        B, S, E = tokens.shape
        NH = self.num_attention_heads
        head_dim = self.attention_head_size
        
        q = self.query(tokens) 
        k = self.key(tokens)   
        v = self.value(tokens)
        
        q = q.view(B, S, NH, head_dim).transpose(1, 2)  
        k = k.view(B, S, NH, head_dim).transpose(1, 2) 
        v = v.view(B, S, NH, head_dim).transpose(1, 2)  
        
        # Scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.cfg.attention_probs_dropout_prob if self.training else 0.0,
            scale=None  
        )  # (B, NH, S, head_dim)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, S, E)
        return attention_output
    
class BertSelfOutput(nn.Module):
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, attention_output, input_tensor):
        hidden_states = self.dense(attention_output)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states

class BertAttention(nn.Module):
    # combines BertSelfAttention + BertSelfOutput
    def __init__(self, config: DistilledBertConfig):
        super().__init__()  
        self.cfg = config
        self.self = BertSelfAttention(self.cfg)  
        self.output = BertSelfOutput(self.cfg) 

    def forward(self, hidden_states):
        self_attention_output = self.self(hidden_states)
        attention_output = self.output(self_attention_output, hidden_states) 
        
        return attention_output
    
class BertIntermediate(nn.Module):
    # first FFN layer with GELU
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    # second FFN layer + LayerNorm
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states
    
class BertLayer(nn.Module):
    # combines attention + FFN (one transformer block)
    def __init__(self, config: DistilledBertConfig):
        super().__init__()
        self.cfg = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states):

        attention_output = self.attention(hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output