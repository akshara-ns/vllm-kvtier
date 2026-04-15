#pragma once
// Compat shim: provides REGISTER_EXTENSION for torch < 2.7.
// In torch >= 2.7, REGISTER_EXTENSION is defined in torch/csrc/stable/library.h
// and registers the extension with the stable ABI runtime.
// For torch 2.6.0 on V100, we just create a minimal Python module init function
// so `import vllm._C_stable_libtorch` succeeds (no ops are registered, but
// all ops in this extension require SM80+/SM90+ anyway).
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define REGISTER_EXTENSION(name)                                          \
  extern "C" PyMODINIT_FUNC PyInit_##name(void) {                        \
    static struct PyModuleDef _module_def = {                             \
        PyModuleDef_HEAD_INIT, #name, nullptr, -1, nullptr,              \
        nullptr, nullptr, nullptr, nullptr                                \
    };                                                                    \
    return PyModule_Create(&_module_def);                                 \
  }
