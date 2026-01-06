#include "../../common/Norms/layernorm.h"
#include "../../common/utils/weight_loader.h"
#include "../../common/Attention/attention.h"
#include "../../common/Matmul/matmul.h"
#include "../../common/utils/residual.h"
#include "../../common/Activation  Functions/gelu.h"
#include "../../common/Activation  Functions/softmax.h"
#include "../../common/Loss Function/CrossEntropy.h"
#include "../../common/utils/tokenizer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
using namespace std;

struct GPT2Config
{
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
};

string load_dataset(const string& filepath) {
    ifstream file(filepath);
    if(!file.is_open()) {
        string alt_path1 = "../" + filepath;
        file.open(alt_path1);
        if(!file.is_open()) {
            string alt_path2 = "../../" + filepath;
            file.open(alt_path2);
            if(!file.is_open()) {
                cerr << "Error: Could not open dataset file: " << filepath << endl;
                cerr << "Tried: " << filepath << ", " << alt_path1 << ", " << alt_path2 << endl;
                return "";
            }
        }
    }
    stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

vector<int> get_batch_tokens(const vector<int>& tokens, int batch_idx, int B, int T, int total_tokens) {
    vector<int> batch;
    int start_idx = batch_idx * B * T;
    for(int i = 0; i < B * T; i++) {
        int idx = (start_idx + i) % total_tokens;
        batch.push_back(tokens[idx]);
    }
    return batch;
}

int main() {
    string dataset_path = "data/input.txt";
    string dataset_text = load_dataset(dataset_path);
    if(dataset_text.empty()) {
        cerr << "Failed to load dataset" << endl;
        return 1;
    }
    
    Tokenizer tokenizer;
    
    string pretrained_vocab_json = "data/tokenizer/vocab.json";
    string pretrained_vocab_txt = "data/vocab.txt";
    
    ifstream test_file(pretrained_vocab_json);
    if(!test_file.is_open()) {
        pretrained_vocab_json = "../" + pretrained_vocab_json;
        test_file.open(pretrained_vocab_json);
        if(!test_file.is_open()) {
            pretrained_vocab_json = "../../" + string("data/tokenizer/vocab.json");
        }
    }
    test_file.close();
    
    test_file.open(pretrained_vocab_txt);
    if(!test_file.is_open()) {
        pretrained_vocab_txt = "../" + pretrained_vocab_txt;
        test_file.open(pretrained_vocab_txt);
        if(!test_file.is_open()) {
            pretrained_vocab_txt = "../../" + string("data/vocab.txt");
        }
    }
    test_file.close();
    
    tokenizer.load_pretrained_vocab_json(pretrained_vocab_json);
    
    if(!tokenizer.get_is_pretrained()) {
        tokenizer.load_pretrained_vocab(pretrained_vocab_txt);
    }
    
    if(!tokenizer.get_is_pretrained()) {
        tokenizer.build_vocab_from_text(dataset_text);
    }
    
    int actual_vocab_size = tokenizer.get_vocab_size();
    vector<int> tokens = tokenizer.encode(dataset_text);
    
    WeightInitializer w;
    GPT2Config config;
    config.max_seq_len = 1024;
    config.vocab_size = actual_vocab_size;
    config.padded_vocab_size = ((actual_vocab_size + 63) / 64) * 64;
    config.num_layers = 12;
    config.num_heads = 12;
    config.channels = 768;
    
    int B = 4;
    int T = 64;
    int C = config.channels;
    int NH = config.num_heads;
    int V = config.vocab_size;
    int Vp = config.padded_vocab_size;
    int num_layers = config.num_layers;
    
    int FF_DIM = 4 * C;
    
    Attention attention;
    Matmul matmul;
    ResidualConnection residual;
    Gelu gelu;
    Softmax softmax;
    CrossEntropy crossentropy;
    
    float* token_embedding_weight = w.Xavier_init(Vp * C, Vp, C);
    float* pos_embedding_weight = w.Xavier_init(config.max_seq_len * C, config.max_seq_len, C);
    
    vector<float*> attn_qkv;      
    vector<float*> attn_preattn; 
    vector<float*> attn_scores;    
    vector<float*> attn_out;       
    vector<float*> attn_residual;  
    vector<float*> ln1_out;       
    vector<float*> ln1_mean;       
    vector<float*> ln1_rstd;     
    
    vector<float*> ff1_out;       
    vector<float*> ff1_gelu;    
    vector<float*> ff2_out;        
    vector<float*> ff_residual;    
    vector<float*> ln2_out;        
    vector<float*> ln2_mean;     
    vector<float*> ln2_rstd;     

    vector<float*> attn_qkv_weight;    
    vector<float*> attn_qkv_bias;      
    vector<float*> attn_proj_weight;   
    vector<float*> attn_proj_bias;     
    vector<float*> ln1_weight;         
    vector<float*> ln1_bias;          
    vector<float*> ff1_weight;         
    vector<float*> ff1_bias;           
    vector<float*> ff2_weight;         
    vector<float*> ff2_bias;          
    vector<float*> ln2_weight;        
    vector<float*> ln2_bias;          
    
    vector<LayerNorm*> ln1_layers;
    vector<LayerNorm*> ln2_layers;
    
    for(int l = 0; l < num_layers; l++) {
        attn_qkv.push_back(w.Zero_init(B * T * 3 * C));
        attn_preattn.push_back(w.Zero_init(B * NH * T * T));
        attn_scores.push_back(w.Zero_init(B * NH * T * T));
        attn_out.push_back(w.Zero_init(B * T * C));
        attn_residual.push_back(w.Zero_init(B * T * C));
        ln1_out.push_back(w.Zero_init(B * T * C));
        ln1_mean.push_back(w.Zero_init(B * T));
        ln1_rstd.push_back(w.Zero_init(B * T));
        
        ff1_out.push_back(w.Zero_init(B * T * FF_DIM));
        ff1_gelu.push_back(w.Zero_init(B * T * FF_DIM));
        ff2_out.push_back(w.Zero_init(B * T * C));
        ff_residual.push_back(w.Zero_init(B * T * C));
        ln2_out.push_back(w.Zero_init(B * T * C));
        ln2_mean.push_back(w.Zero_init(B * T));
        ln2_rstd.push_back(w.Zero_init(B * T));
        
        attn_qkv_weight.push_back(w.Xavier_init(C * 3 * C, C, 3 * C));
        attn_qkv_bias.push_back(w.Zero_init(3 * C));
        attn_proj_weight.push_back(w.Xavier_init(C * C, C, C));
        attn_proj_bias.push_back(w.Zero_init(C));
        ln1_weight.push_back(w.One_init(C));
        ln1_bias.push_back(w.Zero_init(C));
        ff1_weight.push_back(w.Xavier_init(C * FF_DIM, C, FF_DIM));
        ff1_bias.push_back(w.Zero_init(FF_DIM));
        ff2_weight.push_back(w.Xavier_init(FF_DIM * C, FF_DIM, C));
        ff2_bias.push_back(w.Zero_init(C));
        ln2_weight.push_back(w.One_init(C));
        ln2_bias.push_back(w.Zero_init(C));
        
        ln1_layers.push_back(new LayerNorm(ln1_weight[l], ln1_bias[l], C));
        ln2_layers.push_back(new LayerNorm(ln2_weight[l], ln2_bias[l], C));
    }
    
    float* final_ln_out = w.Zero_init(B * T * C);
    float* final_ln_mean = w.Zero_init(B * T);
    float* final_ln_rstd = w.Zero_init(B * T);
    float* final_ln_weight = w.One_init(C);
    float* final_ln_bias = w.Zero_init(C);
    LayerNorm final_ln(final_ln_weight, final_ln_bias, C);
    
    float* logits = w.Zero_init(B * T * Vp);
    float* probs = w.Zero_init(B * T * Vp);
    float* losses = w.Zero_init(B * T);
    float* logits_weight = token_embedding_weight;
    float* logits_bias = w.Zero_init(Vp);
    
    int num_batches = (tokens.size() - 1) / (B * T);
    int num_epochs = 1;
    int max_iterations = 10;
    
    std::cout << "Training started" << std::endl;
    
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        for(int iter = 0; iter < max_iterations && iter < num_batches; iter++) {
            
            vector<int> batch_tokens = get_batch_tokens(tokens, iter, B, T, tokens.size());
            int* token_ids = new int[B * T];
            int* targets = new int[B * T];
            
            for(int i = 0; i < B * T; i++) {
                token_ids[i] = batch_tokens[i];
                if(i < B * T - 1) {
                    targets[i] = batch_tokens[i + 1];
                } else {
                    int next_batch_start = ((iter + 1) * B * T) % tokens.size();
                    targets[i] = tokens[next_batch_start];
                }
                if(targets[i] >= Vp) targets[i] = targets[i] % Vp;
            }
            
            float* token_emb = w.Zero_init(B * T * C);
            float* pos_emb = w.Zero_init(B * T * C);
            float* x = w.Zero_init(B * T * C);
            
            for(int b = 0; b < B; b++) {
                for(int t = 0; t < T; t++) {
                    int token_id = token_ids[b * T + t];
                    if(token_id >= Vp) token_id = token_id % Vp;
                    for(int c = 0; c < C; c++) {
                        token_emb[b * T * C + t * C + c] = token_embedding_weight[token_id * C + c];
                        pos_emb[b * T * C + t * C + c] = pos_embedding_weight[t * C + c];
                        x[b * T * C + t * C + c] = token_emb[b * T * C + t * C + c] + pos_emb[b * T * C + t * C + c];
                    }
                }
            }
            
            float* current_x = x;
            
            for(int l = 0; l < num_layers; l++) {
                ln1_layers[l]->forward(ln1_out[l], ln1_mean[l], ln1_rstd[l], current_x, B, T);
                matmul.forward(attn_qkv[l], ln1_out[l], attn_qkv_weight[l], attn_qkv_bias[l], B, T, C, 3 * C);
                attention.forward(attn_out[l], attn_preattn[l], attn_scores[l], attn_qkv[l], B, T, C, NH);
                matmul.forward(attn_residual[l], attn_out[l], attn_proj_weight[l], attn_proj_bias[l], B, T, C, C);
                residual.forward(current_x, attn_residual[l], ln1_out[l], B * T * C);
                current_x = ln1_out[l];
                
                ln2_layers[l]->forward(ln2_out[l], ln2_mean[l], ln2_rstd[l], current_x, B, T);
                matmul.forward(ff1_out[l], ln2_out[l], ff1_weight[l], ff1_bias[l], B, T, C, FF_DIM);
                gelu.forward(ff1_gelu[l], ff1_out[l], B * T * FF_DIM);
                matmul.forward(ff2_out[l], ff1_gelu[l], ff2_weight[l], ff2_bias[l], B, T, FF_DIM, C);
                residual.forward(current_x, ff2_out[l], ln2_out[l], B * T * C);
                current_x = ln2_out[l];
            }
            
            final_ln.forward(final_ln_out, final_ln_mean, final_ln_rstd, current_x, B, T);
            
            for(int b = 0; b < B; b++) {
                for(int t = 0; t < T; t++) {
                    for(int v = 0; v < Vp; v++) {
                        float val = logits_bias[v];
                        for(int c = 0; c < C; c++) {
                            val += final_ln_out[b * T * C + t * C + c] * token_embedding_weight[v * C + c];
                        }
                        logits[b * T * Vp + t * Vp + v] = val;
                    }
                }
            }
            
            softmax.forward(probs, logits, B, T, V, Vp);
            crossentropy.forward(losses, probs, targets, B, T, Vp);
            
            float total_loss = 0.0f;
            for(int i = 0; i < B * T; i++) {
                total_loss += losses[i];
            }
            float avg_loss = total_loss / (B * T);
            
            std::cout << avg_loss << std::endl;
            
            delete[] token_ids;
            delete[] targets;
            delete[] token_emb;
            delete[] pos_emb;
            delete[] x;
        }
    }
    
    delete[] token_embedding_weight;
    delete[] pos_embedding_weight;
    
    for(int l = 0; l < num_layers; l++) {
        delete[] attn_qkv[l];
        delete[] attn_preattn[l];
        delete[] attn_scores[l];
        delete[] attn_out[l];
        delete[] attn_residual[l];
        delete[] ln1_out[l];
        delete[] ln1_mean[l];
        delete[] ln1_rstd[l];
        delete[] ff1_out[l];
        delete[] ff1_gelu[l];
        delete[] ff2_out[l];
        delete[] ff_residual[l];
        delete[] ln2_out[l];
        delete[] ln2_mean[l];
        delete[] ln2_rstd[l];
        
        delete[] attn_qkv_weight[l];
        delete[] attn_qkv_bias[l];
        delete[] attn_proj_weight[l];
        delete[] attn_proj_bias[l];
        delete[] ln1_weight[l];
        delete[] ln1_bias[l];
        delete[] ff1_weight[l];
        delete[] ff1_bias[l];
        delete[] ff2_weight[l];
        delete[] ff2_bias[l];
        delete[] ln2_weight[l];
        delete[] ln2_bias[l];
        
        delete ln1_layers[l];
        delete ln2_layers[l];
    }
    
    delete[] final_ln_out;
    delete[] final_ln_mean;
    delete[] final_ln_rstd;
    delete[] final_ln_weight;
    delete[] final_ln_bias;
    
    delete[] logits;
    delete[] probs;
    delete[] losses;
    delete[] logits_bias;
    
    return 0;
}