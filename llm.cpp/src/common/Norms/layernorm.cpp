#include <iostream>
#include <cmath>

void layernorm_forward(float* out,float* mean, float* rstd, float* inp, float* weight
                        ,float* bias,int B,int T,int C){
    float eps = 1e-5f;
    for(int b = 0; b < B; b++){
        for(int t = 0; t<T;t++){

            // similar to stride calculation
            float* x = inp + b*(T*C) + t*C;
            float m = 0.0f;
            // for mean
            for(int i = 0;i<C;i++){
                m += x[i]; 
            }
            m = m/C;
            // for variance
            float v = 0.0f;
            for(int i = 0;i < C;i++){
                float xshift = x[i] - m;
                v += xshift*xshift;
            }
            v = v/C;

            float s = 1.0f / sqrtf(v+eps);

            // out_bt = the pointer for the location
            float* out_bt = out + b*(T*C) + t*C;

            for(int i = 0; i < C;i++){
                float n = (s * (x[i]-m));
                float o = n * weight[i] + bias[i];
                // we are updating the value at that location.
                out_bt[i] = o;
            }

            // for caching.
            mean[b*T + t] = m;
            rstd[b*T + t] = s;
        }
    }
}


void layernorm_backward(float* dinp, float* dweight, float* dbias, 
                        float* dout, float* inp, float* weight,float* mean,
                        float* rstd, int B, int T, int C){
    for(int b = 0;b < B;b++){
        for(int t = 0; t < T;t++){

            // getting all the locations of the values to be update
            float* dout_bt = dout + b * (T * C) + t * C;
            float* inp_bt = inp + b * (T * C) + t * C;
            float* dinp_bt = dinp + b * (T * C) + t * C;
            float mean_bt = mean[b * T+ t];
            float rstd_bt = rstd[b * T+ t];

            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;


            // we are recompting the layernorm forward again here but it is a tradeoff we are doing to 
            // save memory.
            for(int i = 0; i<C;i++){
                float norm_bti = (inp_bt[i]-mean_bt)*rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i]; // dL/dx'_i
                dnorm_mean += dnorm_i; 
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}