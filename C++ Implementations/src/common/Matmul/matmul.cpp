#include<cstdio>
#include<cstdlib>

void matmul_naive_forward(float* out, const float* inp,const float* weight,
                            const float* bias, int B, int T, int C, int OC){
    for(int b = 0; b < B;b++){
        for(int t = 0; t < T; t++){
            int bt = b*T+t;
            for(int o = 0; o < OC; o++){
                float val = (bias != NULL)? bias[o]:0.0f;
                for(int i = 0; i < C; i++){
                    val += inp[bt*C+i]*weight[o*C+i];
                }
                out[bt*OC+o] = val;
            }
        }
    }
}



// for backward: we need to compute dinp, dweight, dbias
// dinp[bt, i] += sum_o(dout[bt,o]*weight[o,i])
// dweight[o,t] += sum_bt(dout[bt,o]*inp[bt,i])
// dbias[o] += sum_bt(dout[bt,o])


void matmul_naive_backward(float* dinp, float* dweight, float* dbias, const float* dout,
                            const float* inp, const float* weight, int B, int T, int C, int OC){
    // for dinp
    for(int b = 0;b< B; b++){
        for(int t = 0;t<T;t++){
            const float* dout_bt = dout + b*(T*OC) + t*OC;
            float* dinp_bt = dinp + b*(T*C) + t*C;
            for(int o = 0; o<OC;o++){
                const float* wrow = weight+ o*C;
                float d = dout_bt[o];
                for(int i = 0; i<C;i++){
                    dinp_bt[i] += wrow[i]*d;
                }
            }
        }
    }

    // for dweight and dbias
    for(int o = 0; o<OC; o++){
        for(int b = 0; b<B; b++){
            for(int t = 0; t<T;t++){
                const float* dout_bt = dout + b*(T*OC)+ t*OC;
                const float* inp_bt =  inp + b*(T*C)+ t*C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if(dbias != NULL){
                    dbias[o]+=d;
                }
                for(int i = 0; i<C;i++){
                   dwrow[i] += inp_bt[i]*d;
                }
            }
        }
    }
}



