// 256-bit kernels. Each 64-byte packed group is read as two halves (columns 0-7 and 8-15).
//
// AVX2: vpmaddubsw multiplies u8 x s8 pairs and adds neighbours into saturating int16, then
// vpmaddwd against ones widens to int32. Like intgemm's AVX2 path, a pair sum can saturate at
// 32767; that only happens when both activations and weights sit near their extremes.
//
// AVX-VNNI (Alder Lake, Sapphire Rapids, Zen 5 without AVX-512): the VEX vpdpbusd does the
// same without saturation in one instruction.
//
// Compiled twice: as-is for AVX2 (-mavx2 -mfma) and with RESEAM_AVXVNNI (-mavxvnni).
#include <immintrin.h>

#include "epilogue.h"

namespace reseam::mt::kernels {
namespace {

#ifdef RESEAM_AVXVNNI
constexpr int kMaxMR = 6;  // 12 accumulators + 2 weights + 1 activation of 16 registers
inline __m256i dot4(__m256i acc, __m256i a, __m256i b, __m256i) { return _mm256_dpbusd_avx_epi32(acc, a, b); }
#else
constexpr int kMaxMR = 4;  // 8 accumulators + 2 weights + activation + ones + temp
inline __m256i dot4(__m256i acc, __m256i a, __m256i b, __m256i ones) {
  return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
}
#endif

template <int MR>
inline void tile(const uint8_t *A, int lda, const int8_t *block, int K, int32_t *out /* MR x 16 */) {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i lo[MR], hi[MR];
  for (int r = 0; r < MR; ++r) lo[r] = hi[r] = _mm256_setzero_si256();
  const int groups = K / 4;
  for (int g = 0; g < groups; ++g) {
    const __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(block + size_t(g) * 64));
    const __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(block + size_t(g) * 64 + 32));
    for (int r = 0; r < MR; ++r) {
      const __m256i a = _mm256_set1_epi32(*reinterpret_cast<const int32_t *>(A + size_t(r) * lda + g * 4));
      lo[r] = dot4(lo[r], a, b0, ones);
      hi[r] = dot4(hi[r], a, b1, ones);
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + r * 16), lo[r]);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + r * 16 + 8), hi[r]);
  }
}

}  // namespace

#ifdef RESEAM_AVXVNNI
void gemmAvxVnni(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
#else
void gemmAvx2(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
#endif
  alignas(32) int32_t acc[kMaxMR * 16];
  for (int nb = n0 / kBlockN; nb < n1 / kBlockN; ++nb) {
    const int8_t *block = B.block(nb);
    for (int m = 0; m < M; m += kMaxMR) {
      const int mr = std::min(kMaxMR, M - m);
      const uint8_t *a = A + size_t(m) * lda;
      switch (mr) {
#ifdef RESEAM_AVXVNNI
        case 6: tile<6>(a, lda, block, B.K, acc); break;
        case 5: tile<5>(a, lda, block, B.K, acc); break;
#endif
        case 4: tile<4>(a, lda, block, B.K, acc); break;
        case 3: tile<3>(a, lda, block, B.K, acc); break;
        case 2: tile<2>(a, lda, block, B.K, acc); break;
        default: tile<1>(a, lda, block, B.K, acc); break;
      }
      applyEpilogue(acc, 16, m, mr, nb * kBlockN, kBlockN, B, ep);
    }
  }
}

namespace {

// Horizontal sums of 8 vectors: lane j of the result is the sum of a[j]'s 8 lanes.
inline __m256i reduce8(const __m256i *a) {
  __m256i p[4], q[2];
  for (int i = 0; i < 4; ++i)
    p[i] = _mm256_add_epi32(_mm256_unpacklo_epi32(a[2 * i], a[2 * i + 1]), _mm256_unpackhi_epi32(a[2 * i], a[2 * i + 1]));
  for (int i = 0; i < 2; ++i)
    q[i] = _mm256_add_epi32(_mm256_unpacklo_epi64(p[2 * i], p[2 * i + 1]), _mm256_unpackhi_epi64(p[2 * i], p[2 * i + 1]));
  // q[i]: 128-bit half h holds half h's partial sums of words 4i..4i+3.
  return _mm256_add_epi32(_mm256_permute2x128_si256(q[0], q[1], 0x20), _mm256_permute2x128_si256(q[0], q[1], 0x31));
}

template <int KC>  // K / 32
void argmaxRows(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                const float *bias, float &bestScore, int32_t &bestId) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256 vscale = _mm256_set1_ps(scale);
  float best = bestScore;
  int32_t bestWord = bestId;
  alignas(32) int32_t tail[8];
  for (int i = 0; i < n; i += 8) {
    const int32_t *group = ids + i;
    if (n - i < 8) {  // pad with the last word: a repeat never beats its first occurrence
      for (int j = 0; j < 8; ++j) tail[j] = ids[std::min(i + j, n - 1)];
      group = tail;
    }
    __m256i acc[8];
    for (int j = 0; j < 8; ++j) acc[j] = _mm256_setzero_si256();
    for (int c = 0; c < KC; ++c) {
      const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + c * 32));
      for (int j = 0; j < 8; ++j)
        acc[j] = dot4(acc[j], xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rows + size_t(group[j]) * K + c * 32)), ones);
    }
    const __m256i vid = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(group));
    const __m256 v = _mm256_fmadd_ps(_mm256_cvtepi32_ps(reduce8(acc)), vscale, _mm256_i32gather_ps(bias, vid, 4));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    const float m = _mm_cvtss_f32(m4);
    if (m > best) {
      const int lane = __builtin_ctz(_mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_set1_ps(m), _CMP_EQ_OQ)));
      best = m;
      bestWord = group[lane];
    }
  }
  bestScore = best;
  bestId = bestWord;
}

}  // namespace

#ifdef RESEAM_AVXVNNI
void argmaxAvxVnni(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                   const float *bias, float &bestScore, int32_t &bestId) {
#else
void argmaxAvx2(const uint8_t *x, const int8_t *rows, int K, const int32_t *ids, int n, float scale,
                const float *bias, float &bestScore, int32_t &bestId) {
#endif
  switch (K) {
    case 256: return argmaxRows<8>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case 384: return argmaxRows<12>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    case 512: return argmaxRows<16>(x, rows, K, ids, n, scale, bias, bestScore, bestId);
    default: return argmaxReference(x, rows, K, ids, n, scale, bias, bestScore, bestId);
  }
}

}  // namespace reseam::mt::kernels
