#include <torch/extension.h>

void cache_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                double, double, bool);
void dynamic_cache_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                        at::Tensor, at::Tensor, at::Tensor, bool);
at::Tensor attention_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                          int64_t, double, int64_t);
std::vector<at::Tensor> delta_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    double, double, double, bool, bool, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cache", &cache_cuda);
  m.def("dynamic_cache", &dynamic_cache_cuda);
  m.def("attention", &attention_cuda);
  m.def("delta", &delta_cuda);
}
