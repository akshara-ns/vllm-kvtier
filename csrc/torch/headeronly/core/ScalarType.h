#pragma once
// Compat shim: torch/headeronly/ was introduced in torch 2.7.
// For torch < 2.7 (e.g. 2.6+cu124 on V100), forward to the c10 headers.
#include <c10/core/ScalarType.h>

// Provide torch::headeronly::ScalarType alias so code using the qualified name
// (e.g. torch::headeronly::ScalarType::Int) compiles with torch 2.6.
namespace torch {
namespace headeronly {
using ScalarType = c10::ScalarType;
}  // namespace headeronly
}  // namespace torch
