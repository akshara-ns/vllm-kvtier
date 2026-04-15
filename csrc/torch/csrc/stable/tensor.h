#pragma once
// Compat shim: torch::stable::Tensor was introduced in torch 2.7.
// For torch < 2.7, provide a thin subclass of at::Tensor that adds the
// stable-ABI-specific get_device_index() method.
#include <ATen/Tensor.h>
#include <c10/core/Device.h>

namespace torch {
namespace stable {

struct Tensor : public at::Tensor {
  using at::Tensor::Tensor;

  // Conversion constructors from at::Tensor (needed when returning at::Tensor
  // from ops that produce results as at::Tensor).
  Tensor(const at::Tensor& t) : at::Tensor(t) {}      // NOLINT
  Tensor(at::Tensor&& t) : at::Tensor(std::move(t)) {}  // NOLINT

  // stable ABI method not present on at::Tensor.
  int32_t get_device_index() const {
    return static_cast<int32_t>(at::Tensor::device().index());
  }
};

}  // namespace stable
}  // namespace torch
