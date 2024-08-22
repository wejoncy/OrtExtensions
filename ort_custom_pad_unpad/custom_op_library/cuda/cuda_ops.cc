// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include <assert.h>
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "cuda_context.h"
#include "onnxruntime_lite_custom_op.h"
#include <map>
#include <iostream>
#include <vector>


//#include <cuda.h>
//#include <cuda_runtime.h>

using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \
  if (!(cond)) {                   \
    throw std::runtime_error(msg); \
  }

namespace Cuda {

void AddPadding(const Ort::Custom::CudaContext& cuda_ctx,
               const Ort::Custom::Tensor<Ort::Float16_t>& hidden_states,
               const Ort::Custom::Tensor<int64_t>& attention_mask,
               Ort::Custom::Tensor<Ort::Float16_t>& Z) {
  auto input_shape = attention_mask.Shape();
  auto hidden_size = hidden_states.Shape()[2];
  CUSTOM_ENFORCE(cuda_ctx.cuda_stream, "failed to fetch cuda stream");
  CUSTOM_ENFORCE(cuda_ctx.cudnn_handle, "failed to fetch cudnn handle");
  CUSTOM_ENFORCE(cuda_ctx.cublas_handle, "failed to fetch cublas handle");
  auto z_raw = Z.Allocate({input_shape[0], input_shape[1], hidden_size});
  std::vector<int64_t> host_attn_mask(input_shape[0]*input_shape[1], 0);
  cudaMemcpy (host_attn_mask.data(), attention_mask.Data(), input_shape[0]* input_shape[1]*sizeof(int64_t), cudaMemcpyDeviceToHost );
  std::vector<int64_t>host_attn_mask_lens(input_shape[0], 0);
  size_t pre_tokens = 0;
  for (size_t i=0;i<input_shape[0];i++){
    for (size_t j=0;j<input_shape[1] && host_attn_mask[i *input_shape[1]+ j] == 1;j++){
      host_attn_mask_lens[i]++;
    }
    cudaMemcpyAsync(z_raw+i*input_shape[1]*hidden_size, hidden_states.Data()+pre_tokens*hidden_size, host_attn_mask_lens[i]*sizeof(Ort::Float16_t)*hidden_size, cudaMemcpyDeviceToDevice, cuda_ctx.cuda_stream);
    pre_tokens += host_attn_mask_lens[i];
  }
}

void RemovePadding(const Ort::Custom::CudaContext& cuda_ctx,
               const Ort::Custom::Tensor<int64_t>& input_ids,
               const Ort::Custom::Tensor<int64_t>& attention_mask,
               Ort::Custom::Tensor<int64_t>& Z,
               Ort::Custom::Tensor<int64_t>& Z1
               ) {
  auto input_shape = input_ids.Shape();
  CUSTOM_ENFORCE(cuda_ctx.cuda_stream, "failed to fetch cuda stream");
  CUSTOM_ENFORCE(cuda_ctx.cudnn_handle, "failed to fetch cudnn handle");
  CUSTOM_ENFORCE(cuda_ctx.cublas_handle, "failed to fetch cublas handle");
  std::vector<int64_t> host_attn_mask(input_shape[0]*input_shape[1], 0);
  cudaMemcpy (host_attn_mask.data(), attention_mask.Data(), input_shape[0]*input_shape[1]*sizeof(int64_t), cudaMemcpyDeviceToHost );
  std::vector<int64_t>host_attn_mask_lens(input_shape[0], 0);
  int sum_of_tokens = 0;
  for (size_t i=0;i<input_shape[0];i++){
    for (size_t j=0;j<input_shape[1] && host_attn_mask[i*input_shape[1]+j] == 1;j++){
      host_attn_mask_lens[i]++;
      sum_of_tokens++;
    }
  }
  
  size_t pre_tokens = 0;
  auto z_raw = Z.Allocate({1, sum_of_tokens});
  for (size_t i=0;i<input_shape[0];i++){
    cudaMemcpyAsync(z_raw+pre_tokens, input_ids.Data()+i*input_shape[1], host_attn_mask_lens[i]*sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_ctx.cuda_stream);
    pre_tokens += host_attn_mask_lens[i];
  }

  std::vector<int64_t> position_ids(sum_of_tokens, 0);
  int ind=0;
  for (size_t i=0;i<host_attn_mask_lens.size();i++){
    for(size_t j=0;j<host_attn_mask_lens[i];j++){
       position_ids[ind++] = j;
    }
  }
  auto z1_raw = Z1.Allocate({1, sum_of_tokens});
  cudaMemcpyAsync(z1_raw, position_ids.data(), sum_of_tokens*sizeof(int64_t), cudaMemcpyHostToDevice, cuda_ctx.cuda_stream);

}

void RegisterOps(Ort::CustomOpDomain& domain) {
  static const std::unique_ptr<OrtLiteCustomOp> c_addpadding{Ort::Custom::CreateLiteCustomOp("AddPadding", "CUDAExecutionProvider", AddPadding)};
  static const std::unique_ptr<OrtLiteCustomOp> c_removepadding{Ort::Custom::CreateLiteCustomOp("RemovePadding", "CUDAExecutionProvider", RemovePadding)};
  domain.Add(c_addpadding.get());
  domain.Add(c_removepadding.get());
}

}  // namespace Cuda
