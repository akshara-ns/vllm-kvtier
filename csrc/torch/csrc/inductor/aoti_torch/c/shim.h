#pragma once
// Compat shim: torch/csrc/inductor/aoti_torch/c/shim.h for torch < 2.7.
// This header provides C-ABI compatible functions for use by extensions that
// need to be ABI-stable across torch versions. For torch < 2.7, we implement
// these directly using the regular c10/CUDA APIs.
#include <cstdint>
#include <c10/cuda/CUDAStream.h>

using AOTITorchError = int;
static const AOTITorchError AOTI_TORCH_SUCCESS = 0;

// Returns the current CUDA stream for the given device as a raw pointer.
// device_index == -1 means the current device.
inline AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream) {
  if (device_index < 0) {
    device_index = static_cast<int32_t>(
        c10::cuda::current_device());
  }
  *ret_stream = static_cast<void*>(
      c10::cuda::getCurrentCUDAStream(
          static_cast<c10::DeviceIndex>(device_index))
          .stream());
  return AOTI_TORCH_SUCCESS;
}

// Check the return code of an AOTI C shim call.
// For torch < 2.7, the inline implementations always return SUCCESS, so
// this is effectively a no-op.
#define TORCH_ERROR_CODE_CHECK(expr) (expr)
