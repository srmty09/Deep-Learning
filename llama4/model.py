import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


# llama-4 config:

@dataclass
class Llama4TextConfig:
    vocab_size:int = -1
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers:int = 48
    num_attention_head: int = 40
    # for kv-cache
    num_key_value_heads: int = 8
    # 5120/40 = 128
    head_dim:int = 128

    # though the model only trained on 4096 max_seq_len but it can generate 32 times more tokens
    max_position_embedding = 4096 * 32

    rms_norm_eps: float = 1e-5
    pad_token_id: int = 2000018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_expert_per_tok: int = 2
    num_local_experts: int =16



@dataclass
class Llama4VisionConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate:int  = 5632
    vision_output_dim: int = 7680
    image_size: int = 448
    patch_size:int = 14
    norm_eps: float = 1e-5
    pixel_shuffle_ratio:float = 0.5
    # for projection from the vision block to the text decoder block
    projector_input_dim: int  = 4096
    projector_output_dim:int = 4096
    projector_dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_theta: int = 10000




# Moe: Mixture of Expert

class Llama4TextExperts(nn.Module):
    def __init__(self,config:Llama4TextConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts,self.hidden_size,2*self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts,self.expert_dim,self.hidden_size))
        self.act_fn = nn.SiLU()

        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)


    def forward(self,hidden_states):
        hidden_states = hidden_states.reshape(self.num_experts,-1,self.hidden_size)
        gate_up = torch.bmm(hidden_states,self.gate_up_proj)


        gate,up = gate_up.chunk(2,dim=-1)
        gated = up*self.act_fn(gate)

        next_state = torch.bmm(gated,self.down_proj)
        
        next_state = next_state.view(-1,self.hidden_size)
        return next_state

class Llama4TextMLP(nn.Module):
    def __init__(self,config:Llama4TextConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.up_proj = nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.down_proj = nn.Linear(config.intermediate_size,config.hidden_size,bias = False)

        self.act_fn = nn.SiLU()

    def forward(self,x):
        gated = self.act_fn(self.gate_proj(x))*self.up_proj(x)
        return self.down_proj(gated)
    


class Llama4TextMoe(nn.Module):
    def __init__(self,config:Llama4TextConfig):
        """
        here we are processing the hidden_state to be passed to which expert.
        We take the hidden_states and compute the score for the each router and assign the corresponding embedding to the router.

        """
        super().__init__()
        self.top_k = config.num_expert_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.shared_expert = Llama4TextMLP(config)

        # the router is a linear projection from hidden_size to the number of experts
        self.router = nn.Linear(config.hidden_size,config.num_local_experts,bias=False)


    
    def forward(self,hidden_states):

        batch_size,seq_len,embed_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1,embed_dim)

        # shape: (16,16) here we have 16 tokens and we have 16 experts 
        router_logits = self.router(hidden_states)      

        # step-1 each expert will get all the tokens
        tokens_per_expert = batch_size*seq_len

        # step-2 get the top k expert for each token
        # we are choosing the top_k logit value produced by the router! so dim = -1
        router_top_value, router_indices = torch.topk(router_logits,self.top_k,dim=-1)

        # step-3 create a matrix of -infinities
        router_scores = torch.full_like(router_logits,float("-inf")).scatter_(dim=1,index=router_indices,src=router_top_value).transpose(0,1)

        router_indices = torch.arange(tokens_per_expert,device=hidden_states.device).unsqueeze(0).expand(router_scores.size(0),-1)
        
        router_indices = router_indices.reshape(-1,1).expand(-1,self.hidden_dim)

        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_in = torch.gather(
            input=hidden_states,
            dim = 0,
            index=router_indices
        )

        router_in = router_in*router_scores.reshape(-1,1)

        # all this that we are passing the to the expert
        router_out = self.experts(router_in)

        # shared expert:
        shared_expert_out = self.shared_expert(hidden_states)

        router_out = router_out.reshape(self.num_experts,-1,self.hidden_dim).sum(dim=0)
        next_state = shared_expert_out+router_out     

        return next_state

