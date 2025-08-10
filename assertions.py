import torch

import cuda_softmax_kernel
from triton_kernels.article1 import triton_fused_softmax
from triton_kernels.article2 import triton_softmax_v2
from triton_kernels.article3 import triton_online_softmax, triton_online_softmax_hybrid

# Article 1

x = torch.randn(128, 2**12, device='cuda')
expected_result = torch.softmax(x, dim=-1)

triton_result = triton_fused_softmax(x)
assert torch.allclose(triton_result, expected_result)

# Our implementation is only valid for matrix of width <= 2**12
cuda_result = cuda_softmax_kernel.softmax_cuda_v1(x)
assert torch.allclose(cuda_result, expected_result)

# Article 2

x = torch.randn(128, 2**16, device='cuda')
expected_result = torch.softmax(x, dim=-1)

triton_result = triton_softmax_v2(x)
assert torch.allclose(triton_result, expected_result)

cuda_result = cuda_softmax_kernel.softmax_cuda_multi_block_v1(x)
assert torch.allclose(cuda_result, expected_result)

cuda_result2 = cuda_softmax_kernel.softmax_cuda_multi_block_v2(x)
assert torch.allclose(cuda_result2, expected_result)

# Article 3

x = torch.randn(128, 2**18, device='cuda')
expected_result = torch.softmax(x, dim=-1)

triton_result = triton_online_softmax(x)
assert torch.allclose(triton_result, expected_result)

triton_result = triton_online_softmax_hybrid(x)
assert torch.allclose(triton_result, expected_result)

cuda_result = cuda_softmax_kernel.softmax_cuda_online_v1(x)
assert torch.allclose(cuda_result, expected_result)
