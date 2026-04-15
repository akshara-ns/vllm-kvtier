#pragma once
// Compat shim: torch/headeronly/util/shim_utils.h was introduced in torch 2.7.
// It provides lightweight headeronly versions of TORCH_CHECK etc.
// Since we always link against libtorch, map to the real c10 implementations.
#include <c10/util/Exception.h>

#ifndef STD_TORCH_CHECK
#define STD_TORCH_CHECK TORCH_CHECK
#endif
