#include "../../common/Norms/layernorm.h"
#include<iostream>

struct GPT2Config
{
    int max_seq_len; // max sequence length, 1024
    int vocab_size; // vocab size, 50257
    int padded_vocab_size; // padded to
    int num_layers; // number of layers 12
    int num_heads; // number of heads in attention 12
    int channels; // number of channels 768
};



int main() {
    // int C = 768;  
    // float* weight = new float[C];
    // float* bias = new float[C];
    
    // for(int i = 0; i < C; i++) {
    //     weight[i] = 1.0f;
    //     bias[i] = 0.0f;
    // }
    
    // LayerNorm ln(weight, bias, C);
    GPT2Config config;
    config.channels = 768;  
    std::cout<<config.channels<<std::endl;
    return 0;
}