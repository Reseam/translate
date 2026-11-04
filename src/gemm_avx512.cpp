// AVX-512 VNNI kernel: vpdpbusd multiplies 4 u8 activations by 4 s8 weights and accumulates
// into one int32 lane, 64 multiply-adds per instruction. Register tile: MR rows x 64 columns
// (4 packed blocks), so MR * 4 accumulators (24 for MR = 6) + 4 weight registers.
// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni.
#include <immintrin.h>

#include "epilogue.h"

namespace reseam::mt::kernels {
namespace {

constexpr int kMaxMR = 6;
constexpr int kNR = 4;  // packed blocks per register tile

template <int MR, int NB>
inline void tile(const uint8_t *A, int lda, const int8_t *const *blocks, int K, int32_t *out /* MR x 64 */) {
  __m512i acc[MR][NB];
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < NB; ++j) acc[r][j] = _mm512_setzero_si512();
  const int groups = K / 4;
  for (int g = 0; g < groups; ++g) {
    __m512i b[NB];
    for (int j = 0; j < NB; ++j) b[j] = _mm512_load_si512(blocks[j] + size_t(g) * 64);
    for (int r = 0; r < MR; ++r) {
      const __m512i a = _mm512_set1_epi32(*reinterpret_cast<const int32_t *>(A + size_t(r) * lda + g * 4));
      for (int j = 0; j < NB; ++j) acc[r][j] = _mm512_dpbusd_epi32(acc[r][j], a, b[j]);
    }
  }
  for (int r = 0; r < MR; ++r)
    for (int j = 0; j < NB; ++j) _mm512_storeu_si512(out + r * 64 + j * 16, acc[r][j]);
}

template <int NB>
inline void rows(const uint8_t *A, int lda, int mr, const int8_t *const *blocks, int K, int32_t *out) {
  switch (mr) {
    case 6: return tile<6, NB>(A, lda, blocks, K, out);
    case 5: return tile<5, NB>(A, lda, blocks, K, out);
    case 4: return tile<4, NB>(A, lda, blocks, K, out);
    case 3: return tile<3, NB>(A, lda, blocks, K, out);
    case 2: return tile<2, NB>(A, lda, blocks, K, out);
    default: return tile<1, NB>(A, lda, blocks, K, out);
  }
}

}  // namespace

void gemmAvx512Vnni(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
  alignas(64) int32_t acc[kMaxMR * 64];
  const int b0 = n0 / kBlockN, b1 = n1 / kBlockN;
  // Columns outside, rows inside: one 64-column weight panel (K * 64 bytes, 24 KB at K = 384)
  // stays in L1 while every row tile streams past it.
  for (int nb = b0; nb < b1; nb += kNR) {
    const int nbk = std::min(kNR, b1 - nb);
    const int8_t *blocks[kNR];
    for (int j = 0; j < nbk; ++j) blocks[j] = B.block(nb + j);
    for (int m = 0; m < M; m += kMaxMR) {
      const int mr = std::min(kMaxMR, M - m);
      const uint8_t *a = A + size_t(m) * lda;
      switch (nbk) {
        case 4: rows<4>(a, lda, mr, blocks, B.K, acc); break;
        case 3: rows<3>(a, lda, mr, blocks, B.K, acc); break;
        case 2: rows<2>(a, lda, mr, blocks, B.K, acc); break;
        default: rows<1>(a, lda, mr, blocks, B.K, acc); break;
      }
      applyEpilogue(acc, 64, m, mr, nb * kBlockN, nbk * kBlockN, B, ep);
    }
  }
}

namespace {

// Horizontal sums of 16 vectors at once: lane j of the result is the sum of a[j]'s 16 lanes.
// A transpose-and-add tree: 32-bit pairs, 64-bit pairs, then 128-bit lanes (45 ops for 16).
inline __m512i reduce16(const __m512i *a) {
  __m512i p[8], q[4], r[2];
  for (int i = 0; i < 8; ++i)
    p[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(a[2 * i], a[2 * i + 1]), _mm512_unpackhi_epi32(a[2 * i], a[2 * i + 1]));
  for (int i = 0; i < 4; ++i)
    q[i] = _mm512_add_epi32(_mm512_unpacklo_epi64(p[2 * i], p[2 * i + 1]), _mm512_unpackhi_epi64(p[2 * i], p[2 * i + 1]));
  // q[i]: 128-bit lane L holds lane L's partial sums of words 4i..4i+3.
  for (int i = 0; i < 2; ++i)
    r[i] = _mm512_add_epi32(_mm512_shuffle_i32x4(q[2 * i], q[2 * i + 1], 0x88), _mm512_shuffle_i32x4(q[2 * i], q[2 * i + 1], 0xDD));
  return _mm512_add_epi32(_mm512_shuffle_i32x4(r[0], r[1], 0x88), _mm512_shuffle_i32x4(r[0], r[1], 0xDD));
}

template <int KC>  // K / 64
void argmaxRows(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                const float *bias, float &bestScore, int32_t &bestId) {
  __m512i xv[KC];
  for (int c = 0; c < KC; ++c) xv[c] = _mm512_loadu_si512(x + c * 64);
  const __m512 vscale = _mm512_set1_ps(scale);
  float best = bestScore;
  int32_t bestWord = bestId;
  alignas(64) int32_t tail[16];
  for (int i = 0; i < n; i += 16) {
    const int32_t *group = ids + i;
    if (n - i < 16) {  // pad with the last word: a repeat never beats its first occurrence
      for (int j = 0; j < 16; ++j) tail[j] = ids[std::min(i + j, n - 1)];
      group = tail;
    }
    __m512i acc[16];
    for (int j = 0; j < 16; ++j) {
      const int8_t *w = rows + size_t(group[j]) * K;
      acc[j] = _mm512_dpbusd_epi32(_mm512_setzero_si512(), xv[0], _mm512_loadu_si512(w));
      for (int c = 1; c < KC; ++c) acc[j] = _mm512_dpbusd_epi32(acc[j], xv[c], _mm512_loadu_si512(w + c * 64));
    }
    const __m512i vid = _mm512_loadu_si512(group);
    const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(reduce16(acc)), vscale, _mm512_i32gather_ps(vid, bias, 4));
    const float m = _mm512_reduce_max_ps(v);
    if (m > best) {
      const int lane = __builtin_ctz(_mm512_cmp_ps_mask(v, _mm512_set1_ps(m), _CMP_EQ_OQ));
      best = m;
      bestWord = group[lane];
    }
  }
  bestScore = best;
  bestId = bestWord;
}

}  // namespace

void argmaxAvx512Vnni(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                      const float *bias, float &bestScore, int32_t &bestId) {
  switch (K) {
    case 256: return argmaxRows<4>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case 384: return argmaxRows<6>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case 512: return argmaxRows<8>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    default: return argmaxReference(x, rows, K, ids, n, scale, bias, bestScore, bestId);
  }
}

}  // namespace reseam::mt::kernels
