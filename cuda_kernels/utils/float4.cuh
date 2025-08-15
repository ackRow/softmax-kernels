#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ float max_float4(float4 v) {
    float max_val = fmaxf(v.x, v.y);
    max_val = fmaxf(max_val, v.z);
    max_val = fmaxf(max_val, v.w);
    return max_val;
}

__device__ __forceinline__ float4 stable_exp_float4(float4 v, float max_val) {
    return make_float4(__expf(v.x - max_val),
                       __expf(v.y - max_val),
                       __expf(v.z - max_val),
                       __expf(v.w - max_val));
}

__device__ __forceinline__ float sum_float4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

__device__ __forceinline__ float4 divide_float4(float4 v, float denominator) {
    return make_float4(v.x / denominator,
                       v.y / denominator,
                       v.z / denominator,
                       v.w / denominator);
}

__device__ __forceinline__ float4 full_float4(float v) {
    return make_float4(v, v, v, v);
}
