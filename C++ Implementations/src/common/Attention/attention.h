#ifndef ATTENTION_H
#define ATTENTION_H

void attention_forward(float* out, float*preattn, float* attn, float* inp,
                         int B, int T, int C, int NH);
void attention_backward(float* inp, float* dpreattn, float* dattn, float* dout,
                        float* dinp,float* attn, int B, int T, int C, int NH);         

class Attention{
private:
public:
    void forward(float* out, float*preattn, float* attn, float* inp,
                         int B, int T, int C, int NH){
        attention_forward(out,preattn,attn,inp,B,T,C,NH);
    }
    void backward(float* inp, float* dpreattn, float* dattn, float* dout,
                        float* dinp,float* attn, int B, int T, int C, int NH){
       attention_backward(inp,dpreattn,dattn,dout,dinp,attn,B,T,C,NH);
    }
    
};


#endif ATTENTION_H