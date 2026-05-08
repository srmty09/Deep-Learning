# flow will be to have two different model one cross attention model and other one is the distill bert
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class DBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state: (batch, seq_len, 768)
        return outputs.last_hidden_state


class CrossAttention(nn.Module):
    def __init__(self, num_queries, q_dim, d_model, num_heads):
        super().__init__()
        
        # learnable param queries
        self.queries = nn.Parameter(torch.randn(num_queries,q_dim))
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.d_model = d_model

        self.w_q = nn.Linear(q_dim,d_model)
        self.w_k = nn.Linear(d_model,d_model) 
        self.w_v = nn.Linear(d_model,d_model)
        self.out = nn.Linear(d_model,q_dim)


    def forward(self,image_emb):
        B = image_emb.shape[0]
        Np = image_emb.shape[1]
        Nq = self.queries.shape[0]

        # queries (num_queries,q_dim) -> (Batch_size,num_queries,q_dim)
        Q = self.queries.unsqueeze(0).expand(B,-1,-1)

        Q = self.w_q(Q) # shape: (Batch_size,num_queries,d_model)
        K = self.w_k(image_emb) # shape: (Batch_size,num_patches,d_model)
        V = self.w_v(image_emb)  # shape: (Batch_size,num_patches,d_model)

        
        Q = Q.view(B,Nq,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(B,Np,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(B,Np,self.num_heads,self.head_dim).transpose(1,2)

        scale = self.head_dim**-0.5
        score = torch.matmul(Q,K.transpose(-2,-1)) * scale
        attn = torch.softmax(score,dim=-1)
        out = torch.matmul(attn,V)

        out = out.transpose(1,2).contiguous().view(B,Nq,self.d_model)

        return self.out(out)




image_emb = torch.randn(2,197,768)
model = CrossAttention(32,256,768,8)
print(model(image_emb).shape)





#
#
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
#
# text = "i am a boy!"
# encoder = tokenizer(
#         text,
#         return_tensors="pt",
#         padding = True,
#         truncation = True,
#         max_length = 512
#         )
#
# model = DBertModel()
# outputs = model(
#         encoder['input_ids'],
#         encoder['attention_mask']
#         )
#
# print(encoder["input_ids"].shape)
# print(outputs)
# print(outputs.shape)
