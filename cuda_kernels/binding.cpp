#include <torch/extension.h>

torch::Tensor softmax_cuda_v1(torch::Tensor x);
torch::Tensor softmax_cuda_multi_block_v1(torch::Tensor x);
torch::Tensor softmax_cuda_multi_block_v2(torch::Tensor x);
torch::Tensor softmax_cuda_online_v1(torch::Tensor x);
torch::Tensor softmax_cuda_online_v2(torch::Tensor x);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("softmax_cuda_v1", &softmax_cuda_v1, "Softmax (Baseline)");
  m.def("softmax_cuda_multi_block_v1", &softmax_cuda_multi_block_v1, "Softmax (Multi-block v1)");
  m.def("softmax_cuda_multi_block_v2", &softmax_cuda_multi_block_v2, "Softmax (Multi-block v2)");
  m.def("softmax_cuda_online_v1", &softmax_cuda_online_v1, "Softmax (Online v1)");
  m.def("softmax_cuda_online_v2", &softmax_cuda_online_v2, "Softmax (Online v2)");
}