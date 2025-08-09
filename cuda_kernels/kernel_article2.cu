#include <assert.h>
#include <torch/extension.h>

#include "utils/constants.h"
#include "utils/float4.cuh"
#include "utils/exceptions.h"
#include "utils/reduce.cuh"


template <typename T>
__global__ void
__launch_bounds__(1024, 4)
softmax_kernel_multi_block_v1(T* __restrict__ input, T* __restrict__ output, int n_cols, int n_rows) {
    const int block_size = blockDim.y;
    const int num_warps = CEIL_DIV(block_size, WARP_SIZE);
    __shared__ float reduction[WARP_SIZE];
    
    const int scaled_width = n_cols / 4;
    const int thread_index = threadIdx.y;
    // Check boundary
    if (thread_index >= scaled_width)
        return;

    // Handle multiple rows if the number of rows is above grid size
    for (int row_index = blockIdx.x*blockDim.x + threadIdx.x; row_index < n_rows; row_index += gridDim.x) {
        auto in_vector = reinterpret_cast<const float4*>(&input[row_index * n_cols]);
        
        // 1. Max reduction over a row
        float max_val = -FLT_MAX;
        for (int col_index = thread_index; col_index < scaled_width; col_index += block_size) {
            max_val = fmaxf(max_val, max_float4(in_vector[col_index]));
        }
        max_val = block_wide_max(num_warps, thread_index, max_val, reduction);

        // 2. Sum reduction of the exponentiated row
        float denominator = 0.0f;
        for (int col_index = thread_index; col_index < scaled_width; col_index += block_size) {
            denominator += sum_float4(stable_exp_float4(in_vector[col_index], max_val));
        }
        denominator = block_wide_sum(num_warps, thread_index, denominator, reduction);

        // 3. Compute and store the result of softmax
        auto out_vector = reinterpret_cast<float4*>(&output[row_index * n_cols]);
        for (int col_index = thread_index; col_index < scaled_width; col_index += block_size) {   
            auto numerator = stable_exp_float4(in_vector[col_index], max_val);
            out_vector[col_index] = divide_float4(numerator, denominator);
        }
    }
}

torch::Tensor softmax_cuda_multi_block_v1(torch::Tensor in)
{
    const int n_rows = in.size(0);
    const int n_cols = in.size(1);

    PY_ASSERT(n_cols % 4 == 0); // Assume the input is fully transformable to float4 for simplicity
    
    auto out = torch::empty_like(in);

    int block_dim_y = std::min(n_cols / 4, MAX_BLOCK_SIZE);
    int grid_dim_x = std::min(n_rows, MAX_GRID_SIZE); // 1 block is processing at least 1 entire row

    dim3 block_size(1, block_dim_y, 1);
    dim3 grid_size(grid_dim_x, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "softmax_kernel_multi_block_v1", ([&] {
        softmax_kernel_multi_block_v1<scalar_t><<<grid_size, block_size>>>(
            in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n_cols, n_rows
        );
    }));

    return out;
}


template <typename T, int num_warps, int max_n_cols>
__global__ void
__launch_bounds__(512, 4)
softmax_kernel_multi_block_v2(T* __restrict__ input, T* __restrict__ output, int n_cols, int n_rows) {
    constexpr int block_size = num_warps * WARP_SIZE;
    constexpr int scaled_max_width = max_n_cols / 4;
    constexpr int n_values = CEIL_DIV(scaled_max_width, block_size);

    __shared__ float reduction[WARP_SIZE];

    const int scaled_width = n_cols / 4;
    const int thread_index = threadIdx.y;

    // Handle multiple rows if the number of rows is above grid size
    for (int row_index = blockIdx.x*blockDim.x + threadIdx.x; row_index < n_rows; row_index += gridDim.x) {
        auto in_vector = reinterpret_cast<const float4*>(&input[row_index * n_cols]);
        auto out_vector = reinterpret_cast<float4*>(&output[row_index * n_cols]);
        
        // 0. Load values from in_vector once
        float4 row_values[n_values];
        #pragma unroll n_values
        for (int i = 0; i < n_values; ++i) {
            int col_index = thread_index + i * block_size;
            row_values[i] = (col_index < scaled_width) ? in_vector[col_index] : full_float4(-CUDART_INF_F);
        }

        // 1. Max reduction over a row
        float max_val = -CUDART_INF_F;
        #pragma unroll n_values
        for (int i = 0; i < n_values; ++i) {
            max_val = fmaxf(max_val, max_float4(row_values[i]));
        }
        max_val = block_wide_max(num_warps, thread_index, max_val, reduction);

        // 2. Sum reduction of the exponentiated row
        float denominator = 0.0f;
        #pragma unroll n_values
        for (int i = 0; i < n_values; ++i) {
            denominator += sum_float4(stable_exp_float4(row_values[i], max_val));
        }
        denominator = block_wide_sum(num_warps, thread_index, denominator, reduction);

        // 3. Compute and store the result of softmax
        #pragma unroll n_values
        for (int i = 0; i < n_values; ++i) {
            int col_index = thread_index + i * block_size;
            if (col_index < scaled_width) {
                float4 numerator = stable_exp_float4(row_values[i], max_val);
                out_vector[col_index] = divide_float4(numerator, denominator);
            }
        }
    }
}

template <int num_warps, int max_n_cols>
torch::Tensor launch_softmax_cuda_multi_block_v2(torch::Tensor in)
{
    const int n_rows = in.size(0);
    const int n_cols = in.size(1);

    PY_ASSERT(n_cols % 4 == 0); // Assume the input is fully transformable to float4 for simplicity
    
    auto out = torch::empty_like(in);

    constexpr int block_dim_y = num_warps * WARP_SIZE;
    int grid_dim_x = std::min(n_rows, MAX_GRID_SIZE); // 1 block is processing at least 1 entire row

    dim3 block_size(1, block_dim_y, 1);
    dim3 grid_size(grid_dim_x, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "softmax_kernel_multi_block_v2", ([&] {
        softmax_kernel_multi_block_v2<scalar_t, num_warps, max_n_cols><<<grid_size, block_size>>>(
            in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n_cols, n_rows
        );
    }));

    return out;
}

torch::Tensor softmax_cuda_multi_block_v2(torch::Tensor in) {
    const int width = in.size(1);

    if (width <= 1024)
        return launch_softmax_cuda_multi_block_v2<4, 1024>(in);

    if (width <= 2048)
        return launch_softmax_cuda_multi_block_v2<8, 2048>(in);

    if (width <= 4096)
        return launch_softmax_cuda_multi_block_v2<16, 4096>(in);

    if (width <= 8192)
        return launch_softmax_cuda_multi_block_v2<16, 8192>(in);

    if (width <= 16384)
        return launch_softmax_cuda_multi_block_v2<16, 16384>(in);

    if (width <= 32768)
        return launch_softmax_cuda_multi_block_v2<16, 32768>(in);

    if (width <= 65536)
        return launch_softmax_cuda_multi_block_v2<16, 65536>(in);

    return softmax_cuda_multi_block_v1(in);
}
