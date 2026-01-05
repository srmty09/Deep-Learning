#include <cmath>

void crossentropy_forward(float* losses, float* probs, int* targets, 
                          int B, int T, int Vp){
    for(int b = 0; b<B;b++){
        for(int t = 0; t<T;t++){
            float* probs_bt = probs + b*(T*Vp) + t*Vp;
            int ix = targets[b*T+t];
            losses[b*T+t] = -logf(probs_bt[ix]);
        }
    }
}



void crossentropy_backward(float* dlogits, float* dlosses, float* probs,
                            int* targets, int B, int T,int V, int Vp){
    for(int b = 0; b<B;b++){
        for(int t = 0; t<T;t++){
            float* dlogits_bt = dlogits + b*(T*Vp) + t*Vp;
            float* probs_bt = probs + b*(T*Vp) + t*Vp;
            float dloss = dlosses[b*T+t];
            int ix = targets[b*T+t];

            for(int i = 0; i<V;i++){
                float p = probs_bt[i];
                float tobeconsider = i == ix? 1.0f:0.0f;
                dlogits_bt[i]+=(p-tobeconsider)*dloss;
            }
        }
    }
}
