#include<iostream>

#ifndef RESIDUAL_H
#define RESIDUAL_H

void residual_forward(float* inp1, float* inp2, float* out, int N);
void residual_backward(float* dinp1, float* dinp2, float* dout, int N);

class ResidualConnection{
private:
public:
    void forward(float* inp1, float* inp2, float* out, int N){
        residual_forward(inp1,inp2,out,N);
    }
    void backward(float* dinp1, float* dinp2, float* dout, int N){
        residual_backward(dinp1,dinp2,dout,N);
    }
};

#endif RESIDUAL_H