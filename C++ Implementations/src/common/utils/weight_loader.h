#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>

float* zero_init(size_t size);
float* xavier_init(size_t size, int fan_in, int fan_out);
float* one_init(size_t size);


class WeightInitializer{
private:
public:
    float* Zero_init(size_t size){
        return zero_init(size);
    }
    float* Xavier_init(size_t size, int fan_in, int fan_out){
        return xavier_init(size, fan_in, fan_out);
    }
    float* One_init(size_t size){
        return one_init(size);
    }
};


#endif // WEIGHT_LOADER_H

