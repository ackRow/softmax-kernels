#include <torch/extension.h>

torch::Tensor softmax_cuda_v1(torch::Tensor x);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("softmax_cuda_v1", &softmax_cuda_v1, "Softmax (CUDA)");
}