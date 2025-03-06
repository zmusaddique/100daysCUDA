#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor,>
fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fa2_backward(torch::Tensor dO, torch::Tensor Q, torch::Tensor K,
             torch::Tensor V, torch::Tensor O, torch::Tensor M, torch::Tensor L,
             bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fa2_forward, "Flash attention 2 forward pass");
  m.def("backward", &fa2_backward, "Flash attention 2 backward pass");
}
