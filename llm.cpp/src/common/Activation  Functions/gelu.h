#include <cmath>

#ifndef GELU_H
#define GELU_H

void gelu_forward(float* out, float* inp, int N);
void gelu_backward(float* dinp, float* inp,float* dout, int N);


class Gelu {

public:
    void forward(float* out, float* inp, int N){
        gelu_forward(out,inp,N);
    }
    void backward(float* dinp,float* inp, float* dout, int N){
        gelu_backward(dinp,inp,dout,N);
    }

};


#endif // GELU_H