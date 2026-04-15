#pragma once
// Compat shim: torch/csrc/stable/accelerator.h was introduced in torch 2.7.
// Provides torch::stable::accelerator::DeviceGuard backed by c10::cuda::CUDAGuard.
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace torch {
namespace stable {
namespace accelerator {

class DeviceGuard {
 public:
  explicit DeviceGuard(int32_t device_index)
      : guard_(static_cast<c10::DeviceIndex>(device_index)) {}

 private:
  c10::cuda::CUDAGuard guard_;
};

}  // namespace accelerator
}  // namespace stable
}  // namespace torch
