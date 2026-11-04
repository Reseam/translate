// int8 matrix multiply for the Firefox Translations (Marian "intgemm8 alphas") models.
//
// C[M x N] = A[M x K] * B[K x N], where A holds activations quantized to uint8 with a fixed
// per-layer scale ("shifted": value + 127) and B holds the model's int8 weights. The int32 result
// goes through a fused epilogue instead of being written out:
//
//   value = acc * scale[n] + bias[n]        (bias already includes the -127 * colsum shift fix)
//
// and then, depending on the epilogue: stored as float, ReLU'd and requantized straight into the
// next layer's uint8 input.
//
// B is packed once at load time in a layout every kernel reads natively:
//
//   [N/16 column blocks][K/4 groups][16 columns][4 consecutive k]
//
// One 64-byte group is exactly one AVX-512 VNNI operand (vpdpbusd: 4 u8 x 4 s8 per int32 lane)
// and one row of an AMX B tile; AVX2 reads each group as two 32-byte halves.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace reseam::mt {

constexpr int kBlockN = 16;  // columns per packed block
constexpr int kAlignK = 64;  // K is padded to a multiple of this (one AMX tile row)

struct AlignedFree {
  void operator()(void *p) const;
};
template <class T>
using AlignedPtr = std::unique_ptr<T[], AlignedFree>;
template <class T>
AlignedPtr<T> alignedAlloc(size_t count, size_t alignment = 64);

/// A weight matrix packed for the kernels, with its per-column dequantization. A view: the
/// memory belongs to the model's storage (heap or a memory-mapped cache file) or a PackedOwner.
struct PackedB {
  int K = 0;  // rows of B (input width), a multiple of kAlignK
  int N = 0;  // columns of B (output width), a multiple of kBlockN
  const int8_t *data = nullptr;
  const float *scale = nullptr;  // per column: 1 / (quantMultA * quantMultB)
  const float *bias = nullptr;   // per column: bias - 127 * colsum(B) * scale
  float quantMultA = 0;          // scale applied to activations before this matrix

  const int8_t *block(int nb) const { return data + size_t(nb) * K * kBlockN; }
};

inline int paddedK(int K) { return (K + kAlignK - 1) / kAlignK * kAlignK; }
inline int paddedN(int N) { return (N + kBlockN - 1) / kBlockN * kBlockN; }

/// Packs a K x N int8 matrix into `dst` (paddedK(K) x paddedN(N) packed bytes, zeroed by the
/// caller) starting at column `col0`, reading element (k, n) from src[k * strideK + n * strideN]
/// (row-major: strideK = N, strideN = 1). Columns are filled independently.
void packColumns(const int8_t *src, int K, int N, size_t strideK, size_t strideN, int8_t *dst, int Kpad, int col0);

/// Per-column dequantization for columns [col0, col0 + n) of packed `B` (row count Kpad):
/// scale = 1 / (qA * qB), and the bias with the shift correction folded in.
void dequantColumns(const int8_t *B, int Kpad, int col0, int n, float quantMultA, float quantMultB, const float *bias,
                    float *scaleOut, float *biasOut);

/// A packed matrix that owns its memory (tests and tools; models use their own storage).
struct PackedOwner {
  PackedB view;
  AlignedPtr<int8_t> data;
  std::vector<float> scale, bias;
  PackedOwner(int K, int N);
  void pack(const int8_t *src, int K, int N, size_t strideK, size_t strideN, int col0);
  void dequant(int col0, int n, float quantMultA, float quantMultB, const float *bias);
};

/// What happens to each output value after dequantization.
struct Epilogue {
  enum Kind {
    kStoreFloat,  // out[m * ld + n] = v
    kReluQuant,   // outU8[m * ld + n] = quantize(max(v, 0) * nextQuantMultA)
  } kind = kStoreFloat;

  float *out = nullptr;
  uint8_t *outU8 = nullptr;
  int ld = 0;
  float nextQuantMultA = 0;

  /// The same epilogue for a sub-problem whose row 0 is output row m0.
  Epilogue fromRow(int m0) const {
    Epilogue e = *this;
    if (e.out) e.out += size_t(m0) * e.ld;
    if (e.outU8) e.outU8 += size_t(m0) * e.ld;
    return e;
  }
};

/// Instruction set used for the int8 products, picked once at startup.
enum class Isa { kReference, kAvx2, kAvxVnni, kAvx512Vnni, kAmx };
Isa detectIsa();
const char *isaName(Isa isa);
/// Overrides detection (for tests and benchmarks); returns false if the CPU lacks it.
bool forceIsa(Isa isa);
Isa activeIsa();

/// C = A * B for rows [0, M) and B columns [n0, n1) (multiples of kBlockN), with the epilogue.
/// A is M x B.K uint8, row stride lda (a multiple of 64 bytes for AMX).
void gemm(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
inline void gemm(const uint8_t *A, int lda, int M, const PackedB &B, const Epilogue &ep) {
  gemm(A, lda, M, B, 0, B.N, ep);
}

/// Quantizes float rows to the shifted uint8 form the kernels take:
/// clamp(round_half_even(x * q), -127, 127) + 127. Pads each row with 127 (zero) up to `lda`.
void quantizeRows(const float *x, int ldx, int M, int K, float quantMult, uint8_t *out, int lda);

/// Output layer for one row: the best-scoring word among `ids` (n entries) given the row's
/// quantized hidden state `x` (K uint8, shifted). Word w's weights are `rows + w * K` (int8, the
/// target embedding row), its score acc * scale + bias[w]. Updates bestScore/bestId when a word
/// scores strictly higher, so ties keep the earliest word in `ids`.
void argmaxCandidates(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                      const float *bias, float &bestScore, int32_t &bestId);

// Per-ISA entry points (each compiled with its own target flags).
namespace kernels {
void argmaxReference(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                     const float *bias, float &bestScore, int32_t &bestId);
void argmaxAvx2(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                const float *bias, float &bestScore, int32_t &bestId);
void argmaxAvxVnni(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                   const float *bias, float &bestScore, int32_t &bestId);
void argmaxAvx512Vnni(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                      const float *bias, float &bestScore, int32_t &bestId);
void gemmReference(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
void gemmAvx2(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
void gemmAvxVnni(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
void gemmAvx512Vnni(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
void gemmAmx(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep);
bool amxEnable();  // asks the kernel for AMX tile state; false if unavailable
}  // namespace kernels

}  // namespace reseam::mt
