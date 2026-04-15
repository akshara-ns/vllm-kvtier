#pragma once
// Compat shim: torch/csrc/stable/ops.h was introduced in torch 2.7.
// This header provides stable ABI wrappers for common torch ops.
// For torch < 2.7, this is an empty stub — no ops from this header are
// called directly in vLLM's compat path.
