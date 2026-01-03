#ifndef SOFTMAX_H
#define SOFTMAX_H

void softmax_forward(float* probs, float* logits,int B,
                int T, int V, int Vp);

class softmax{
private:
public:
    void forward(float* probs, float* logits,int B,
                int T, int V, int Vp){
        softmax_forward(probs, logits, B,
                T, V, Vp);
    }
    
};


#endif SOFTMAX_H