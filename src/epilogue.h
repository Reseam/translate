// Epilogue shared by every kernel: included into each kernel's translation unit so it is compiled
// (and auto-vectorized) with that unit's instruction set. Everything here has internal linkage so
// the linker can never swap an AVX-512 copy into the AVX2 path.
#pragma once

#include <algorithm>
#include <cmath>

#include "gemm.h"

namespace reseam::mt {
namespace {

// Applies `ep` to an int32 tile: rows [m0, m0 + rows), columns [n0, n0 + cols) of the output,
// with acc[r * ldAcc + c] holding the product for (m0 + r, n0 + c).
inline void applyEpilogue(const int32_t *acc, int ldAcc, int m0, int rows, int n0, int cols, const PackedB &B,
                          const Epilogue &ep) {
  const float *scale = B.scale + n0;
  const float *bias = B.bias + n0;
  switch (ep.kind) {
    case Epilogue::kStoreFloat:
      for (int r = 0; r < rows; ++r) {
        float *out = ep.out + size_t(m0 + r) * ep.ld + n0;
        const int32_t *a = acc + size_t(r) * ldAcc;
        for (int c = 0; c < cols; ++c) out[c] = float(a[c]) * scale[c] + bias[c];
      }
      break;
    case Epilogue::kReluQuant: {
      const float q = ep.nextQuantMultA;
      for (int r = 0; r < rows; ++r) {
        uint8_t *out = ep.outU8 + size_t(m0 + r) * ep.ld + n0;
        const int32_t *a = acc + size_t(r) * ldAcc;
        for (int c = 0; c < cols; ++c) {
          float v = std::max(float(a[c]) * scale[c] + bias[c], 0.0f) * q;
          v = std::min(std::nearbyint(v), 127.0f);  // v >= 0, so only the top needs clamping
          out[c] = uint8_t(int(v) + 127);
        }
      }
      break;
    }
  }
}

}  // namespace
}  // namespace reseam::mt
