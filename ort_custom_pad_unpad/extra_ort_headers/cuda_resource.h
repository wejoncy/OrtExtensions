// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resource.h"

#define ORT_CUDA_RESOUCE_VERSION 1

enum CudaResource : int {
#ifndef USE_ROCM
  cuda_stream_t = cuda_resource_offset,
#else
  cuda_stream_t = rocm_resource_offset,
#endif
  cudnn_handle_t,
  cublas_handle_t
};
