#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

void crossentropy_forward(float* losses, float* probs, int* targets, 
                          int B, int T, int Vp);
void crossentropy_backward(float* dlogits, float* dlosses, float* probs,
                            int* targets, int B, int T, int V, int Vp);

class CrossEntropy{
private:
public:
    void forward(float* losses, float* probs, int* targets, 
                          int B, int T, int Vp){
        crossentropy_forward(losses,probs,targets,B,T,Vp);
    }
    void backward(float* dlogits, float* dlosses, float* probs,
                            int* targets, int B, int T, int V, int Vp){
        crossentropy_backward(dlogits,dlosses,probs,targets,B,T,V,Vp);
    }
};



#endif CROSSENTROPY_H