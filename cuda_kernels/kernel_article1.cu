#include <assert.h>
#include <torch/extension.h>

#include "utils/constants.h"
#include "utils/float4.cuh"
#include "utils/exceptions.h"
#include "utils/reduce.cuh"


template <typename T>
__global__ void
__launch_bounds__(1024, 4)
softmax_kernel_v1(T* __restrict__ input, T* __restrict__ output, int n_cols, int n_rows) {
    const int num_warps = CEIL_DIV(blockDim.y, WARP_SIZE);
    __shared__ float reduction[WARP_SIZE];
    
    const int scaled_width = n_cols / 4;
    const int col_index = threadIdx.y;
    // Check boundary
    if (col_index >= scaled_width)
        return;

    // Handle multiple rows if the number of rows is above grid size
    for (int row_index = blockIdx.x*blockDim.x + threadIdx.x; row_index < n_rows; row_index += gridDim.x) {
        auto in_vector = reinterpret_cast<const float4*>(&input[row_index * n_cols]);
        
        // 1. Max reduction over a row
        float max_val = max_float4(in_vector[col_index]);
        max_val = block_wide_max(num_warps, col_index, max_val, reduction);

        // 2. Stable exponential of a row
        auto numerator = stable_exp_float4(in_vector[col_index], max_val);

        // 3. Sum reduction of the exponentiated row
        float denominator = sum_float4(numerator);
        denominator = block_wide_sum(num_warps, col_index, denominator, reduction);

        // 4. Compute and store the result of softmax
        auto out_vector = reinterpret_cast<float4*>(&output[row_index * n_cols]);
        out_vector[col_index] = divide_float4(numerator, denominator);
    }
}

torch::Tensor softmax_cuda_v1(torch::Tensor in)
{
    const int n_rows = in.size(0);
    const int n_cols = in.size(1);

    PY_ASSERT(n_cols % 4 == 0); // Assume the input is fully transformable to float4 for simplicity
    PY_ASSERT(n_cols <= 4 * MAX_BLOCK_SIZE); // This implementation expects an entire row to fit in a single block

    auto out = torch::empty_like(in);

    int block_dim_y = n_cols / 4; // 1 thread is processing 4 values
    int grid_dim_x = std::min(n_rows, MAX_GRID_SIZE); // 1 block is processing at least 1 entire row

    dim3 block_size(1, block_dim_y, 1);
    dim3 grid_size(grid_dim_x, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "softmax_kernel_v1", ([&] {
        softmax_kernel_v1<scalar_t><<<grid_size, block_size>>>(
            in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n_cols, n_rows
        );
    }));

    return out;
}
