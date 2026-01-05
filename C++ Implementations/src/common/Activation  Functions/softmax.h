#ifndef SOFTMAX_H
#define SOFTMAX_H

void softmax_forward(float* probs, float* logits,int B,
                int T, int V, int Vp);
void softmax_backward(float* dlogits, const float* dprobs, const float* probs,
                      int B, int T, int V, int Vp);              

class Softmax{
private:
public:
    void forward(float* probs, float* logits,int B,
                int T, int V, int Vp){
        softmax_forward(probs, logits, B,
                T, V, Vp);
    }
    void backward(float* dlogits, const float* dprobs, const float* probs,
                      int B, int T, int V, int Vp){
        softmax_backward(dlogits,dprobs,probs,B,T,V,Vp);
    }
    
};


#endif // SOFTMAX_H