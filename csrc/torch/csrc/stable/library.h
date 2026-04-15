#pragma once
// Compat shim: torch/csrc/stable/library.h was introduced in torch 2.7.
// Maps stable ABI library macros to the regular torch::Library equivalents.
// Since we always link against libtorch, the regular API is available.
#include <torch/library.h>

// Map STABLE_TORCH_LIBRARY_* to regular TORCH_LIBRARY_* macros.
// The only difference in the API is the parameter type (torch::stable::Library
// vs torch::Library), but with our alias below they are the same.
#define STABLE_TORCH_LIBRARY_FRAGMENT(ns, var) TORCH_LIBRARY_FRAGMENT(ns, var)
#define STABLE_TORCH_LIBRARY_IMPL(ns, key, var) TORCH_LIBRARY_IMPL(ns, key, var)

// TORCH_BOX wraps a fn ptr for the stable C ABI calling convention.
// With regular libtorch registration, pass the function pointer directly.
#define TORCH_BOX(fn) fn

namespace torch {
namespace stable {
  using Library = torch::Library;
}  // namespace stable
}  // namespace torch
