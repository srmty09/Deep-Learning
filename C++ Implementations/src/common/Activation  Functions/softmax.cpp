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