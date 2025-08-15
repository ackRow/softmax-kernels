#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#define WARP_SIZE 32

__device__ __forceinline__ float block_wide_max(
    const int num_warps, const int thread_index, float local_max, float* shared_reduction
) {
    const int warp_index = thread_index / WARP_SIZE;

    // Warp-level reduction
    for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, stride));
    }

    // Store warp result in shared memory
    if (thread_index % WARP_SIZE == 0)
        shared_reduction[warp_index] = local_max;
    __syncthreads();

    // Only first warp participates in final reduction
    if (warp_index == 0 && thread_index < num_warps) {
        local_max = shared_reduction[thread_index];
        unsigned mask = __activemask();
        for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, stride));
        }
    }

    // Final result broadcasted via shared memory
    if (thread_index == 0)
        shared_reduction[0] = local_max;
    __syncthreads();

    return shared_reduction[0];
}

__device__ __forceinline__  float block_wide_sum(
    const int num_warps, const int thread_index, float local_exp, float* shared_reduction
) {
    const int warp_index = thread_index / WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        local_exp += __shfl_xor_sync(0xFFFFFFFF, local_exp, stride);
    }

    if (thread_index % WARP_SIZE == 0)
        shared_reduction[warp_index] = local_exp;
    __syncthreads();

    if (warp_index == 0 && thread_index < num_warps) {
        local_exp = shared_reduction[thread_index];
        unsigned mask = __activemask();
        for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
            local_exp += __shfl_xor_sync(mask, local_exp, stride);
        }
    }

    if (thread_index == 0)
        shared_reduction[0] = local_exp;
    __syncthreads();

    return shared_reduction[0];
}

__device__ __forceinline__ void online_softmax_update(
    float *current_max, float *current_divisor, float incoming_max, float incoming_divisor
) {
    float max_value = fmaxf(*current_max, incoming_max);
    *current_divisor = *current_divisor * __expf(*current_max - max_value) + \
                        incoming_divisor * __expf(incoming_max - max_value);
    *current_max = max_value;
}

__device__ __forceinline__  void block_wide_online_softmax(
    const int num_warps, const int thread_index,
    float* local_max, float* local_div,
    float* shared_max, float* shared_div
) {
    const int warp_index = thread_index / WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        float incoming_max = __shfl_xor_sync(0xFFFFFFFF, *local_max, stride);
        float incoming_div = __shfl_xor_sync(0xFFFFFFFF, *local_div, stride);
        online_softmax_update(local_max, local_div, incoming_max, incoming_div);
    }

    if (thread_index % WARP_SIZE == 0) {
        shared_max[warp_index] = *local_max;
        shared_div[warp_index] = *local_div;
    }
    __syncthreads();

    if (warp_index == 0 && thread_index < num_warps) {
        *local_max = shared_max[thread_index];
        *local_div = shared_div[thread_index];

        unsigned mask = __activemask();
        for (int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
            float incoming_max = __shfl_xor_sync(mask, *local_max, stride);
            float incoming_div = __shfl_xor_sync(mask, *local_div, stride);
            online_softmax_update(local_max, local_div, incoming_max, incoming_div);
        }
    }

    if (thread_index == 0) {
        shared_max[0] = *local_max;
        shared_div[0] = *local_div;
    }
    __syncthreads();
    *local_max = shared_max[0];
    *local_div = shared_div[0];
}