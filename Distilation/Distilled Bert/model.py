import torch
import torch.nn as nn
from dataclasses import dataclass


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
    

