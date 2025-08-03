import torch

from .triton_online_softmax import triton_online_softmax, triton_online_softmax_v2

x = torch.randn(128, 2**18, device='cuda')
expected_result = torch.softmax(x, dim=-1)

result = triton_online_softmax(x)
assert torch.allclose(result, expected_result)

triton_online_result = triton_online_softmax_v2(x)
assert torch.allclose(result, expected_result)
