#include <torch/extension.h>

#include "utils/constants.h"
#include "utils/float4.cuh"
#include "utils/exceptions.h"
#include "utils/reduce.cuh"


__device__ void online_softmax_divisor_update(
    float *current_max, float *current_divisor, float incoming_max, float incoming_divisor
) {
    float max_value = fmaxf(*current_max, incoming_max);
    float divisor = *current_divisor * __expf(*current_max - max_value) + incoming_divisor * __expf(incoming_max - max_value);

    *current_max = max_value;
    *current_divisor = divisor;
}

template <typename T, int num_warps>
__global__ void softmax_kernel_online_warp_reduction(T* __restrict__ input, T* __restrict__ output, int w, int h) {
    constexpr int block_dim_y = num_warps * WARP_SIZE;

    const int thread_index = threadIdx.y;
    const int warp_index = thread_index / WARP_SIZE;
    __shared__ float shared_max[num_warps];
    __shared__ float shared_div[num_warps];

    for (int row = blockIdx.x*blockDim.x + threadIdx.x; row < h; row += gridDim.x) {

        // Local reduction
        float max_val = -FLT_MAX;
        float divisor = 0.0f;
        
        for (int i = thread_index; i < w; i += block_dim_y) {
            float incoming_max = input[row * w + i];
            online_softmax_divisor_update(&max_val, &divisor, incoming_max, 1.0f);
        }
        
        // Intra-warp reduction
        for(int stride = WARP_SIZE/2; stride >= 1; stride /= 2) {
            float incoming_max = __shfl_xor_sync(0xFFFFFFFF, max_val, stride);
            float incoming_div = __shfl_xor_sync(0xFFFFFFFF, divisor, stride);
            online_softmax_divisor_update(&max_val, &divisor, incoming_max, incoming_div);
        }

        // Inter-warp synchronization
        if (thread_index % WARP_SIZE == 0) {
            shared_max[warp_index] = max_val;
            shared_div[warp_index] = divisor;
        }
        __syncthreads();

        // First warp is reducing inter-warp shared values
        if (warp_index == 0 && thread_index < num_warps) {
            max_val = shared_max[thread_index];
            divisor = shared_div[thread_index];

            unsigned mask = __activemask();
            for(int stride = WARP_SIZE/2; stride >= 1; stride /= 2) {
                float incoming_max = __shfl_xor_sync(mask, max_val, stride);
                float incoming_div = __shfl_xor_sync(mask, divisor, stride);
                
                online_softmax_divisor_update(&max_val, &divisor, incoming_max, incoming_div);
            }
        }

        // Share final values to all threads
        if (thread_index == 0) {
            shared_max[0] = max_val;
            shared_div[0] = divisor;
        }
        __syncthreads();
        max_val = shared_max[0];
        divisor = shared_div[0];
        
        for (int i = thread_index; i < w; i += block_dim_y) {
            output[row * w + i] = __expf(input[row * w + i] - max_val) / divisor;
        }
    }
}

template <int num_warps>
torch::Tensor launch_softmax_online_warp_reduction(torch::Tensor in)
{  
    const int n_rows = in.size(0);
    const int n_cols = in.size(1);

    auto out = torch::empty_like(in);

    constexpr int block_dim_y = num_warps * WARP_SIZE;
    int grid_dim_x = std::min(n_rows, MAX_GRID_SIZE);

    dim3 block_size(1, block_dim_y, 1);
    dim3 grid_size(grid_dim_x, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "online_softmax_v1", ([&] {
        softmax_kernel_online_warp_reduction<scalar_t, num_warps><<<grid_size, block_size>>>(
            in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n_cols, n_rows
        );
    }));

    return out;
}

torch::Tensor softmax_cuda_online_v1(torch::Tensor in) {
    const int w = in.size(1);

    if (w >= 8192)
        return launch_softmax_online_warp_reduction<32>(in);
    else if (w >= 4096)
        return launch_softmax_online_warp_reduction<16>(in);
    else if (w >= 2048)
        return launch_softmax_online_warp_reduction<8>(in);
    else
        return launch_softmax_online_warp_reduction<4>(in);
}