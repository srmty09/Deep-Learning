#include "weight_loader.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

float* zero_init(size_t size){
    float* tensor = new float[size];
    for(size_t t = 0; t<size;t++){
        tensor[t] = 0.0f;
    }
    return tensor;
}

float* xavier_init(size_t size, int fan_in, int fan_out){
    float* tensor = new float[size];
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    for(size_t t = 0; t<size;t++){
        tensor[t] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }
    return tensor;
}


float* one_init(size_t size){
    float* tensor = new float[size];
    for(size_t t = 0; t<size;t++){
        tensor[t] = 1.0f;
    }
    return tensor;
}