import torch

import cuda_softmax_kernel
from triton_tutorial import triton_fused_softmax

x = torch.randn(128, 2**12, device='cuda')
expected_result = torch.softmax(x, dim=-1)

triton_result = triton_fused_softmax(x)
assert torch.allclose(triton_result, expected_result)

# Our implementation is only valid for matrix of width <= 2**12
cuda_result = cuda_softmax_kernel.softmax_cuda_v1(x)
assert torch.allclose(cuda_result, expected_result)
