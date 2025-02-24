#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
fa2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa2_forward", torch::wrap_pybind_function(fa2_forward), "fa2_forward");
}
