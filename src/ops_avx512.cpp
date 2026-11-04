// AVX-512 attention: 16 keys per vector, so a typical post's keys fit in one or two registers.
// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512dq.
#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace reseam::mt::ops {

namespace {

// exp(x) for 16 floats, the same Cephes polynomial as the AVX2 version.
inline __m512 exp16(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f)), _mm512_set1_ps(88.3762626647949f));
  __m512 fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f));
  fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
  const __m512 z = _mm512_mul_ps(x, x);
  __m512 y = _mm512_set1_ps(1.9875691500E-4f);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507E-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073E-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894E-2f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459E-1f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201E-1f));
  y = _mm512_fmadd_ps(y, z, x);
  y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));
  const __m512i e = _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(0x7f));
  return _mm512_mul_ps(y, _mm512_castsi512_ps(_mm512_slli_epi32(e, 23)));
}

template <int HD>
void attention16(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk, int heads,
                 float *out, int ldo, float *p) {
  constexpr int kVec = HD / 16 > 0 ? HD / 16 : 1;  // head width in zmm (48 -> 3, 32 -> 2, 64 -> 4)
  static_assert(HD % 16 == 0, "head width must be a multiple of 16");
  const __m512 scale = _mm512_set1_ps(1.0f / std::sqrt(float(HD)));
  const int chunks = (nk + 15) / 16;
  const __mmask16 tailMask = __mmask16((1u << (nk - (chunks - 1) * 16)) - 1);
  for (int i = 0; i < nq; ++i) {
    for (int h = 0; h < heads; ++h) {
      const float *qh = q + size_t(i) * ldq + h * HD;
      const float *kh = kt + size_t(h) * HD * ldkt;
      for (int c0 = 0; c0 < chunks; c0 += 4) {
        const int nc = std::min(4, chunks - c0);
        __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int d = 0; d < HD; ++d) {
          const __m512 qd = _mm512_set1_ps(qh[d]);
          const float *row = kh + size_t(d) * ldkt + c0 * 16;
          for (int u = 0; u < nc; ++u) acc[u] = _mm512_fmadd_ps(qd, _mm512_loadu_ps(row + u * 16), acc[u]);
        }
        for (int u = 0; u < nc; ++u) _mm512_storeu_ps(p + (c0 + u) * 16, _mm512_mul_ps(acc[u], scale));
      }
      // softmax over the nk real keys (padding lanes masked out)
      __m512 vmax = _mm512_set1_ps(-INFINITY);
      for (int c = 0; c < chunks; ++c)
        vmax = _mm512_mask_max_ps(vmax, c + 1 < chunks ? __mmask16(0xFFFF) : tailMask, vmax, _mm512_loadu_ps(p + c * 16));
      const __m512 vm = _mm512_set1_ps(_mm512_reduce_max_ps(vmax));
      __m512 vsum = _mm512_setzero_ps();
      for (int c = 0; c < chunks; ++c) {
        const __mmask16 m = c + 1 < chunks ? __mmask16(0xFFFF) : tailMask;
        const __m512 e = _mm512_maskz_mov_ps(m, exp16(_mm512_sub_ps(_mm512_loadu_ps(p + c * 16), vm)));
        _mm512_storeu_ps(p + c * 16, e);
        vsum = _mm512_add_ps(vsum, e);
      }
      const __m512 inv = _mm512_set1_ps(1.0f / _mm512_reduce_add_ps(vsum));
      __m512 o[kVec];
      for (int c = 0; c < kVec; ++c) o[c] = _mm512_setzero_ps();
      for (int j = 0; j < nk; ++j) {
        const __m512 w = _mm512_set1_ps(p[j]);
        const float *vh = v + size_t(j) * ldv + h * HD;
        for (int c = 0; c < kVec; ++c) o[c] = _mm512_fmadd_ps(w, _mm512_loadu_ps(vh + c * 16), o[c]);
      }
      float *oh = out + size_t(i) * ldo + h * HD;
      for (int c = 0; c < kVec; ++c) _mm512_storeu_ps(oh + c * 16, _mm512_mul_ps(o[c], inv));
    }
  }
}

}  // namespace

void attentionT512(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk,
                   int heads, int headDim, float *out, int ldo, float *scratch) {
  switch (headDim) {
    case 32: return attention16<32>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    case 48: return attention16<48>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    case 64: return attention16<64>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    default: break;
  }
  // Other head widths: AVX2 path handles them (never reached with Firefox models).
  for (int i = 0; i < nq; ++i)
    for (int h = 0; h < heads; ++h) {
      const float *qh = q + size_t(i) * ldq + h * headDim;
      float maxv = -INFINITY;
      for (int j = 0; j < nk; ++j) {
        float s = 0;
        for (int d = 0; d < headDim; ++d) s += qh[d] * kt[size_t(h * headDim + d) * ldkt + j];
        scratch[j] = s / std::sqrt(float(headDim));
        maxv = std::max(maxv, scratch[j]);
      }
      float sum = 0;
      for (int j = 0; j < nk; ++j) sum += scratch[j] = std::exp(scratch[j] - maxv);
      float *oh = out + size_t(i) * ldo + h * headDim;
      for (int d = 0; d < headDim; ++d) {
        float acc = 0;
        for (int j = 0; j < nk; ++j) acc += scratch[j] * v[size_t(j) * ldv + h * headDim + d];
        oh[d] = acc / sum;
      }
    }
}

}  // namespace reseam::mt::ops
