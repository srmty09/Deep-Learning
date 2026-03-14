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
    num_expert_per_tok: int = 1
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



