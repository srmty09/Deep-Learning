#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../Activation  Functions/softmax.h"
#include "../Matmul/matmul.h"


void attention_forward(float* out, float*preattn, float* attn, float* inp,
                         int B, int T, int C, int NH){
    int C3 = 3*C;
    int hs = C/NH; // head size
    float scaling_factor = 1.0f/sqrtf(hs);
    Softmax act;

    for(int b = 0; b<B; b++){
        for(int t = 0; t<T; t++){
            for(int h = 0; h<NH;h++){
                float* query_t = inp + b*(T*C3)+ t*C3 + h*hs;
                float* preattn_bth = preattn + b*NH*T*T + h*T*T + t*T;
                float* attn_bth = attn + b*NH*T*T + h*T*T + t*T;
                
                // why till t because attention mask.
                for(int t2 = 0; t2<=t; t2++){
                    float* key_t2 = inp + b*(T*C3)+ t2*C3 + h*hs + C;
                    float val = 0.0f;

                    // inner most loop - Query dot Key
                    for(int i = 0; i<hs; i++){
                        val += query_t[i]*key_t2[i];
                    }
                    val *= scaling_factor;
                    preattn_bth[t2] = val;
                }

                // Apply softmax
                act.forward(attn_bth, preattn_bth, 1, 1, t+1,t+1);
                
                // Weighted sum of values
                float* out_bth = out + b*T*C + t*C + h*hs;
                for(int i = 0; i<hs; i++) { out_bth[i] = 0.0f; }
                
                for(int t2 = 0; t2<=t; t2++){
                    float* value_t2 = inp + b*(T*C3) + t2*C3 + h*hs + C*2;
                    float att_weight = attn_bth[t2];
                    for(int i = 0; i<hs; i++){
                        out_bth[i] += att_weight * value_t2[i];
                    }
                }
            }
        }
    }
}



void attention_backward(float* inp, float* dpreattn, float* dattn, float* dout,
                        float* dinp,float* attn, int B, int T, int C, int NH){
    int C3 = C*3;
    int hs = C/NH;
    float scaling_factor = 1.0f/sqrtf(hs);
    Softmax act;

    for(int b = 0; b < B; b++){
        for(int t = 0; t<T;t++){
            for(int h = 0; h<NH;h++){
                float* attn_bth = attn + b*NH*T*T + h*T*T + t*T;
                float* dattn_bth = dattn + b*NH*T*T + h*T*T + t*T;
                float* dpreattn_bth = dpreattn + b*NH*T*T + h*T*T + t*T;
                float* dout_bth = dout + b*T*C + t*C + h*hs;
                
                // Backward through weighted sum of values
                for(int t2 = 0; t2<=t; t2++){
                    float* value_t2 = inp + b*T*C3 + t2*C3 + h*hs + C*2;
                    float* dvalue_t2 = dinp + b*T*C3 + t2*C3 + h*hs + C*2;
                    float att_weight = attn_bth[t2];
                    
                    for(int i = 0; i<hs; i++){
                        // Gradient w.r.t. attention weights
                        dattn_bth[t2] += dout_bth[i] * value_t2[i];
                        // Gradient w.r.t. values
                        dvalue_t2[i] += dout_bth[i] * att_weight;
                    }
                }
                
                // Backward through softmax
                act.backward(dpreattn_bth, dattn_bth, attn_bth, 1, 1, t+1,t+1);
                
                // Backward through Q·K^T
                for(int t2 = 0; t2<=t; t2++){
                    float* key_t2 = inp + b*T*C3 + t2*C3 + h*hs + C;
                    float* dkey_t2 = dinp + b*T*C3 + t2*C3 + h*hs + C;
                    float* dquery_t = dinp + b*T*C3 + t*C3 + h*hs;
                    float* query_t = inp + b*T*C3 + t*C3 + h*hs;
                    
                    float d = dpreattn_bth[t2] * scaling_factor;
                    
                    for(int i = 0; i<hs; i++){
                        // Gradient w.r.t. query
                        dquery_t[i] += key_t2[i] * d;
                        // Gradient w.r.t. key
                        dkey_t2[i] += query_t[i] * d;
                    }
                }
            }
        }
    }
}




