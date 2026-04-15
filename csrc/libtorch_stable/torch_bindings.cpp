// _C_stable_libtorch extension — all ops are CUTLASS SM80+/SM90+ and are not
// available on SM70 (V100). For torch < 2.7 (torch/csrc/stable/ does not
// exist), build an empty module that loads cleanly but registers no ops.
// This satisfies `import vllm._C_stable_libtorch` in cuda.py.
#include "core/registration.h"
REGISTER_EXTENSION(_C_stable_libtorch)
