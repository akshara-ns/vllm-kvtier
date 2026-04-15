#pragma once
// Compat shim: torch/headeronly/ was introduced in torch 2.7.
// For torch < 2.7 (e.g. 2.6+cu124 on V100), forward to the c10 headers.
#include <c10/util/Float8_e4m3fn.h>
