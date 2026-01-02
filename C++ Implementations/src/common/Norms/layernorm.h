#ifndef LAYERNORM_H
#define LAYERNORM_H

void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight,
                       float* bias, int B, int T, int C);

void layernorm_backward(float* dinp, float* dweight, float* dbias, 
                        float* dout, float* inp, float* weight, float* mean,
                        float* rstd, int B, int T, int C);


class LayerNorm {
private:
    float* weight;
    float* bias;
    int C;  
    
public:
    LayerNorm(float* w, float* b, int channels) : weight(w), bias(b), C(channels) {}
    
    void forward(float* out, float* mean, float* rstd, float* inp, int B, int T) {
        layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);
    }
    
    void backward(float* dinp, float* dweight, float* dbias, 
                  float* dout, float* inp, float* mean, float* rstd, int B, int T) {
        layernorm_backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    }
};

#endif // LAYERNORM_H

