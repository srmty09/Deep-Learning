#ifndef MATMUL_NAIVE
#define MATMUL_NAIVE


void matmul_naive_forward(float* out, const float* inp,const float* weight,
                            const float* bias, int B, int T, int C, int OC);

void matmul_naive_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                            const float* inp, const float* weight, int B, int T, int C, int OC);


class matmul{
private:
public:
    void forward(float* out, const float* inp, const float* weight,
                const float* bias, int B, int T, int C, int OC){
                    matmul_naive_forward(out,inp,weight,bias,B,T,C,OC);
                }
    void backward(float* dinp, float* dweight, float* dbias, const float* dout,
                            const float* inp, const float* weight, int B, int T, int C, int OC){
                    matmul_naive_backward(dinp,dweight,dbias,dout,inp,weight,B,T,C,OC);
                            }
};


#endif MATMUL_NAIVE