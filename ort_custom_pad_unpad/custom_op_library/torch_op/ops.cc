#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor AddPadding(
    torch::Tensor input,
    torch::Tensor mask);

torch::Tensor RemovePadding(
    torch::Tensor input,
    torch::Tensor mask);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor AddPadding(
    torch::Tensor input,
    torch::Tensor mask) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  auto seq_len = mask.sum(-1).cpu();
  auto total_seq_len = seq_len.sum().cpu()[0];
  auto output = torch::empty({mask.sizes()[0], mask.sizes()[1], input.size(-1)},
                           mask.Options());
  for (size_t i=0;i<mask.size(0);i++) {
    
  }
  return output;
}

torch::Tensor RemovePadding(
    torch::Tensor input,
    torch::Tensor mask) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return lltm_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("AddPadding", &AddPadding, "AddPadding (CUDA)");
  m.def("RemovePadding", &RemovePadding, "RemovePadding (CUDA)");
}

