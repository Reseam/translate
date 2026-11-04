// Checks the transposed-key attention against the row-major reference implementation.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gemm.h"
#include "ops.h"

using namespace reseam::mt;

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.0f, 2.0f);
  int failures = 0;
  for (Isa isa : {Isa::kAvx2, Isa::kAvx512Vnni}) {
    if (!forceIsa(isa)) continue;
    for (int headDim : {32, 48, 64})
      for (int nk : {1, 5, 8, 15, 16, 17, 31, 33, 64, 128})
        for (int nq : {1, 3}) {
          const int heads = 8, dim = heads * headDim, ld = 3 * dim;
          std::vector<float> qkv(size_t(std::max(nq, nk)) * ld);
          for (auto &v : qkv) v = dist(rng);
          const float *q = qkv.data(), *k = q + dim, *v = q + 2 * dim;
          std::vector<float> want(size_t(nq) * dim), got(size_t(nq) * dim), scratch(ops::keyStride(nk) + 64);
          ops::attention(q, ld, nq, k, ld, v, ld, nk, heads, headDim, want.data(), dim, scratch.data());
          const int ldkt = ops::keyStride(nk);
          std::vector<float> kt(size_t(dim) * ldkt);
          ops::transposeKeys(k, ld, nk, dim, kt.data(), ldkt);
          ops::attentionT(q, ld, nq, kt.data(), ldkt, v, ld, nk, heads, headDim, got.data(), dim, scratch.data());
          float worst = 0;
          for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(want[i] - got[i]));
          if (!(worst < 1e-4f)) {
            std::printf("FAIL isa=%s headDim=%d nk=%d nq=%d maxdiff=%g\n", isaName(isa), headDim, nk, nq, worst);
            ++failures;
          }
        }
  }
  std::printf(failures ? "%d failures\n" : "attention matches\n", failures);
  return failures != 0;
}
