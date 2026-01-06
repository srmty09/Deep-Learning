#include <math.h>

void softmax_forward(float* probs, float* logits,int B,
                int T, int V, int Vp){
    
    for(int b = 0; b<B; b++){
        for(int t = 0; t<T;t++){
            float* logits_bt = logits + b*(T*Vp) + t*Vp;
            float* probs_bt = probs + b*(T*Vp) + t*Vp;

            float maxval = -10000.0f;
            for(int i = 0; i<V;i++){
                if(logits_bt[i]>maxval){
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for(int i = 0; i < V; i++){
                probs_bt[i] = expf(logits_bt[i]-maxval);
                sum  += probs_bt[i];
            }

            for(int i = 0; i<V;i++){
                probs_bt[i]/=sum;
            }

            for(int i = V; i<Vp;i++){
                probs_bt[i] = 0.0f;
            }
        }
    }
}


// dprobs: gradient from next layer (same shape as probs)
// probs: output of softmax forward
// dlogits: output gradient (same shape as logits)
void softmax_backward(float* dlogits, const float* dprobs, const float* probs,
                      int B, int T, int V, int Vp) {

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {

            float* dlogits_bt = dlogits + b*(T*Vp) + t*Vp;
            const float* dprobs_bt = dprobs + b*(T*Vp) + t*Vp;
            const float* probs_bt  = probs  + b*(T*Vp) + t*Vp;

            // dot = sum_j (dprobs_j * probs_j)
            float dot = 0.0f;
            for (int i = 0; i < V; i++) {
                dot += dprobs_bt[i] * probs_bt[i];
            }

            // dlogits_i = probs_i * (dprobs_i - dot)
            for (int i = 0; i < V; i++) {
                dlogits_bt[i] = probs_bt[i] * (dprobs_bt[i] - dot);
            }

            // padded part
            for (int i = V; i < Vp; i++) {
                dlogits_bt[i] = 0.0f;
            }
        }
    }
}
