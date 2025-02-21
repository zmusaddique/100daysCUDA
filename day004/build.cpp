#include <torch/extension.h>

torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

torch::Tensor q_kt_mul(torch::Tensor q, torch::Tensor k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa2_forward", torch::wrap_pybind_function(fa2_forward), "fa2_forward");
  m.def("q_kt_mul", torch::wrap_pybind_function(q_kt_mul), "q_kt_mul");
}
