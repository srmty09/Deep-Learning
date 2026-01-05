#include <iostream>
#include <cmath>

#define GELU_SCALING_FACTOR sqrtf(2.0f/M_PI)
void gelu_forward(float* out, float* inp, int N){
    for(int i = 0; i < N; i++){
        float x = inp[i];
        float cube = 0.044715f*x*x*x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x+cube)));
    }
}
void gelu_backward(float* dinp, float* inp,float* dout, int N){
    for(int i = 0; i<N;i++){
        float x = inp[i];
        float cube = 0.044715f*x*x*x;
        float tanh_arg = GELU_SCALING_FACTOR * (x+cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out*coshf_out);
        float _grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += _grad * dout[i];
    }
}
