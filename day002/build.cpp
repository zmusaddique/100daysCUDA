#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k,
                                  torch::Tensor v);

PYBIND11_MODULE(flash_attention2, m) {
  m.def("flash_attention_fwd", &fa2_forward, "Flash Attention Forward Pass");
}

