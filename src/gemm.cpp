#include "gemm.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "epilogue.h"

namespace reseam::mt {

void AlignedFree::operator()(void *p) const { std::free(p); }

template <class T>
AlignedPtr<T> alignedAlloc(size_t count, size_t alignment) {
  size_t bytes = (count * sizeof(T) + alignment - 1) / alignment * alignment;
  void *p = std::aligned_alloc(alignment, bytes ? bytes : alignment);
  if (!p) throw std::bad_alloc();
  return AlignedPtr<T>(static_cast<T *>(p));
}
template AlignedPtr<char> alignedAlloc<char>(size_t, size_t);
template AlignedPtr<int8_t> alignedAlloc<int8_t>(size_t, size_t);
template AlignedPtr<uint8_t> alignedAlloc<uint8_t>(size_t, size_t);
template AlignedPtr<float> alignedAlloc<float>(size_t, size_t);
template AlignedPtr<int32_t> alignedAlloc<int32_t>(size_t, size_t);

void packColumns(const int8_t *src, int K, int N, size_t strideK, size_t strideN, int8_t *dst, int Kpad, int col0) {
  for (int n = 0; n < N; ++n) {
    const int col = col0 + n;
    int8_t *block = dst + size_t(col / kBlockN) * Kpad * kBlockN;
    const int c = col % kBlockN;
    for (int k = 0; k < K; ++k) block[(size_t(k / 4) * kBlockN + c) * 4 + (k % 4)] = src[k * strideK + n * strideN];
  }
}

void dequantColumns(const int8_t *B, int Kpad, int col0, int n, float quantMultA, float quantMultB, const float *bias,
                    float *scaleOut, float *biasOut) {
  const float s = 1.0f / (quantMultA * quantMultB);
  for (int j = 0; j < n; ++j) {
    const int col = col0 + j;
    const int8_t *block = B + size_t(col / kBlockN) * Kpad * kBlockN;
    int32_t colsum = 0;
    for (int k = 0; k < Kpad; ++k) colsum += block[(size_t(k / 4) * kBlockN + col % kBlockN) * 4 + (k % 4)];
    scaleOut[col] = s;
    // The kernels multiply (a + 127) by b, so 127 * colsum(b) is subtracted back out here.
    biasOut[col] = (bias ? bias[j] : 0.0f) - 127.0f * float(colsum) * s;
  }
}

PackedOwner::PackedOwner(int K, int N) {
  view.K = paddedK(K);
  view.N = paddedN(N);
  data = alignedAlloc<int8_t>(size_t(view.K) * view.N);
  std::memset(data.get(), 0, size_t(view.K) * view.N);
  scale.assign(view.N, 0.0f);
  bias.assign(view.N, 0.0f);
  view.data = data.get();
  view.scale = scale.data();
  view.bias = bias.data();
}

void PackedOwner::pack(const int8_t *src, int K, int N, size_t strideK, size_t strideN, int col0) {
  packColumns(src, K, N, strideK, strideN, data.get(), view.K, col0);
}

void PackedOwner::dequant(int col0, int n, float quantMultA, float quantMultB, const float *b) {
  view.quantMultA = quantMultA;
  dequantColumns(data.get(), view.K, col0, n, quantMultA, quantMultB, b, scale.data(), bias.data());
}

void quantizeRows(const float *x, int ldx, int M, int K, float quantMult, uint8_t *out, int lda) {
  for (int m = 0; m < M; ++m) {
    const float *row = x + size_t(m) * ldx;
    uint8_t *o = out + size_t(m) * lda;
    for (int k = 0; k < K; ++k) {
      float v = std::nearbyint(row[k] * quantMult);
      v = std::min(std::max(v, -127.0f), 127.0f);
      o[k] = uint8_t(int(v) + 127);
    }
    for (int k = K; k < lda; ++k) o[k] = 127;  // padding quantizes zero
  }
}

namespace kernels {

void gemmReference(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
  int32_t acc[kBlockN];
  for (int m = 0; m < M; ++m) {
    for (int nb = n0 / kBlockN; nb < n1 / kBlockN; ++nb) {
      const int8_t *block = B.block(nb);
      for (int c = 0; c < kBlockN; ++c) {
        int32_t s = 0;
        for (int k = 0; k < B.K; ++k) s += int32_t(A[size_t(m) * lda + k]) * block[(size_t(k / 4) * kBlockN + c) * 4 + k % 4];
        acc[c] = s;
      }
      applyEpilogue(acc, kBlockN, m, 1, nb * kBlockN, kBlockN, B, ep);
    }
  }
}

void argmaxReference(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                     const float *bias, float &bestScore, int32_t &bestId) {
  for (int i = 0; i < n; ++i) {
    const int8_t *w = rows + size_t(ids[i]) * K;
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) acc += int32_t(x[k]) * w[k];
    const float v = float(acc) * scale + bias[ids[i]];
    if (v > bestScore) {
      bestScore = v;
      bestId = ids[i];
    }
  }
}

}  // namespace kernels

// ---- Dispatch ------------------------------------------------------------------------------

namespace {

// __builtin_cpu_supports needs literal feature names, hence the macro.
#define RESEAM_CPU_HAS(feature) (__builtin_cpu_init(), __builtin_cpu_supports(feature))

bool supported(Isa isa) {
  switch (isa) {
    case Isa::kReference: return true;
    case Isa::kAvx2: return RESEAM_CPU_HAS("avx2") && RESEAM_CPU_HAS("fma");
    case Isa::kAvxVnni: return RESEAM_CPU_HAS("avx2") && RESEAM_CPU_HAS("avxvnni");
    case Isa::kAvx512Vnni:
      return RESEAM_CPU_HAS("avx512bw") && RESEAM_CPU_HAS("avx512vl") && RESEAM_CPU_HAS("avx512vnni");
    case Isa::kAmx: {
      static const bool ok = RESEAM_CPU_HAS("avx512vnni") && RESEAM_CPU_HAS("amx-int8") && kernels::amxEnable();
      return ok;
    }
  }
  return false;
}

Isa gIsa = Isa::kReference;
bool gDetected = false;

}  // namespace

Isa detectIsa() {
  if (const char *env = std::getenv("RESEAM_MT_ISA")) {
    for (Isa isa : {Isa::kReference, Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512Vnni, Isa::kAmx})
      if (std::string(env) == isaName(isa) && supported(isa)) return isa;
  }
  for (Isa isa : {Isa::kAmx, Isa::kAvx512Vnni, Isa::kAvxVnni, Isa::kAvx2})
    if (supported(isa)) return isa;
  return Isa::kReference;
}

const char *isaName(Isa isa) {
  switch (isa) {
    case Isa::kReference: return "reference";
    case Isa::kAvx2: return "avx2";
    case Isa::kAvxVnni: return "avxvnni";
    case Isa::kAvx512Vnni: return "avx512vnni";
    case Isa::kAmx: return "amx";
  }
  return "?";
}

bool forceIsa(Isa isa) {
  if (!supported(isa)) return false;
  gIsa = isa;
  gDetected = true;
  return true;
}

Isa activeIsa() {
  if (!gDetected) {
    gIsa = detectIsa();
    gDetected = true;
  }
  return gIsa;
}

void argmaxCandidates(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                      const float *bias, float &bestScore, int32_t &bestId) {
  if (n <= 0) return;
  switch (activeIsa()) {
    case Isa::kAmx:
    case Isa::kAvx512Vnni: return kernels::argmaxAvx512Vnni(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case Isa::kAvxVnni: return kernels::argmaxAvxVnni(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case Isa::kAvx2: return kernels::argmaxAvx2(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case Isa::kReference: return kernels::argmaxReference(x, rows, K, ids, n, scale, bias, bestScore, bestId);
  }
}

void gemm(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
  if (M <= 0 || n1 <= n0) return;
  switch (activeIsa()) {
    case Isa::kAmx: return kernels::gemmAmx(A, lda, M, B, n0, n1, ep);
    case Isa::kAvx512Vnni: return kernels::gemmAvx512Vnni(A, lda, M, B, n0, n1, ep);
    case Isa::kAvxVnni: return kernels::gemmAvxVnni(A, lda, M, B, n0, n1, ep);
    case Isa::kAvx2: return kernels::gemmAvx2(A, lda, M, B, n0, n1, ep);
    case Isa::kReference: return kernels::gemmReference(A, lda, M, B, n0, n1, ep);
  }
}

}  // namespace reseam::mt
