#include "ops.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

#include "gemm.h"

namespace reseam::mt::ops {

// AVX-512 variants live in ops_avx512.cpp (compiled with AVX-512 flags).
void attentionT512(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk,
                   int heads, int headDim, float *out, int ldo, float *scratch);

namespace {
bool useAvx512() {
  static const bool yes = activeIsa() == Isa::kAvx512Vnni || activeIsa() == Isa::kAmx;
  return yes;
}
}  // namespace

namespace {

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

inline float hmax(__m256 v) {
  __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_max_ps(s, _mm_movehl_ps(s, s));
  s = _mm_max_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// exp(x) for 8 floats, Cephes polynomial (about 1 ulp over the float range we see).
inline __m256 exp8(__m256 x) {
  const __m256 hi = _mm256_set1_ps(88.3762626647949f), lo = _mm256_set1_ps(-88.3762626647949f);
  x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
  __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  const __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
  __m256i e = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(0x7f));
  return _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32(e, 23)));
}

template <int HD>
inline float dotHead(const float *a, const float *b) {
  __m256 s = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
  for (int i = 8; i < HD; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
  return hsum(s);
}

inline float dotAny(const float *a, const float *b, int n) {
  __m256 s = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
  float r = hsum(s);
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}

template <int HD>
void attentionHeads(const float *q, int ldq, int nq, const float *k, int ldk, const float *v, int ldv, int nk,
                    int heads, int headDim, float *out, int ldo, float *p) {
  const float scale = 1.0f / std::sqrt(float(headDim));
  for (int i = 0; i < nq; ++i) {
    for (int h = 0; h < heads; ++h) {
      const float *qh = q + size_t(i) * ldq + h * headDim;
      float maxv = -INFINITY;
      for (int j = 0; j < nk; ++j) {
        const float *kh = k + size_t(j) * ldk + h * headDim;
        const float s = (HD ? dotHead<HD>(qh, kh) : dotAny(qh, kh, headDim)) * scale;
        p[j] = s;
        maxv = std::max(maxv, s);
      }
      // softmax over the keys
      const __m256 vmax = _mm256_set1_ps(maxv);
      __m256 vsum = _mm256_setzero_ps();
      int j = 0;
      for (; j + 8 <= nk; j += 8) {
        const __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p + j), vmax));
        _mm256_storeu_ps(p + j, e);
        vsum = _mm256_add_ps(vsum, e);
      }
      float sum = hsum(vsum);
      if (j < nk) {
        alignas(32) float tail[8] = {0};
        for (int t = j; t < nk; ++t) tail[t - j] = p[t] - maxv;
        alignas(32) float et[8];
        _mm256_store_ps(et, exp8(_mm256_load_ps(tail)));
        for (int t = j; t < nk; ++t) {
          p[t] = et[t - j];
          sum += et[t - j];
        }
      }
      const float inv = 1.0f / sum;
      // weighted sum of values
      float *o = out + size_t(i) * ldo + h * headDim;
      if constexpr (HD > 0) {
        __m256 acc[HD / 8];
        for (int c = 0; c < HD / 8; ++c) acc[c] = _mm256_setzero_ps();
        for (int jj = 0; jj < nk; ++jj) {
          const __m256 w = _mm256_set1_ps(p[jj] * inv);
          const float *vh = v + size_t(jj) * ldv + h * headDim;
          for (int c = 0; c < HD / 8; ++c) acc[c] = _mm256_fmadd_ps(w, _mm256_loadu_ps(vh + c * 8), acc[c]);
        }
        for (int c = 0; c < HD / 8; ++c) _mm256_storeu_ps(o + c * 8, acc[c]);
      } else {
        for (int d = 0; d < headDim; ++d) o[d] = 0;
        for (int jj = 0; jj < nk; ++jj) {
          const float w = p[jj] * inv;
          const float *vh = v + size_t(jj) * ldv + h * headDim;
          for (int d = 0; d < headDim; ++d) o[d] += w * vh[d];
        }
      }
    }
  }
}

}  // namespace

void addLayerNorm(const float *a, const float *b, int M, int dim, const LayerNorm &norm, float *out) {
  const float *gamma = norm.gamma, *beta = norm.beta;
  for (int m = 0; m < M; ++m) {
    const float *ra = a + size_t(m) * dim, *rb = b ? b + size_t(m) * dim : nullptr;
    float *ro = out + size_t(m) * dim;
    __m256 vs = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
      __m256 x = _mm256_loadu_ps(ra + i);
      if (rb) x = _mm256_add_ps(x, _mm256_loadu_ps(rb + i));
      _mm256_storeu_ps(ro + i, x);
      vs = _mm256_add_ps(vs, x);
    }
    const float mean = hsum(vs) / dim;
    const __m256 vm = _mm256_set1_ps(mean);
    __m256 vq = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
      const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(ro + i), vm);
      vq = _mm256_fmadd_ps(d, d, vq);
    }
    const float inv = 1.0f / std::sqrt(hsum(vq) / dim + 1e-6f);
    const __m256 vi = _mm256_set1_ps(inv);
    for (int i = 0; i < dim; i += 8) {
      const __m256 d = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(ro + i), vm), vi);
      _mm256_storeu_ps(ro + i, _mm256_fmadd_ps(d, _mm256_loadu_ps(gamma + i), _mm256_loadu_ps(beta + i)));
    }
  }
}

void attention(const float *q, int ldq, int nq, const float *k, int ldk, const float *v, int ldv, int nk, int heads,
               int headDim, float *out, int ldo, float *scratch) {
  switch (headDim) {
    case 32: return attentionHeads<32>(q, ldq, nq, k, ldk, v, ldv, nk, heads, headDim, out, ldo, scratch);
    case 48: return attentionHeads<48>(q, ldq, nq, k, ldk, v, ldv, nk, heads, headDim, out, ldo, scratch);
    case 64: return attentionHeads<64>(q, ldq, nq, k, ldk, v, ldv, nk, heads, headDim, out, ldo, scratch);
    default: return attentionHeads<0>(q, ldq, nq, k, ldk, v, ldv, nk, heads, headDim, out, ldo, scratch);
  }
}

void transposeKeys(const float *k, int ldk, int nk, int dim, float *kt, int ldkt) {
  for (int d = 0; d < dim; ++d) {
    float *row = kt + size_t(d) * ldkt;
    for (int j = 0; j < nk; ++j) row[j] = k[size_t(j) * ldk + d];
    for (int j = nk; j < ldkt; ++j) row[j] = 0.0f;
  }
}

namespace {

// Softmax over p[0..nk) in place (p has room for a multiple of 8), returns 1 / sum.
inline float softmax8(float *p, int nk) {
  const int full = nk / 8 * 8;
  __m256 vmax = _mm256_set1_ps(-INFINITY);
  for (int j = 0; j < full; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(p + j));
  float maxv = hmax(vmax);
  for (int j = full; j < nk; ++j) maxv = std::max(maxv, p[j]);
  const __m256 vm = _mm256_set1_ps(maxv);
  __m256 vsum = _mm256_setzero_ps();
  for (int j = 0; j < full; j += 8) {
    const __m256 e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p + j), vm));
    _mm256_storeu_ps(p + j, e);
    vsum = _mm256_add_ps(vsum, e);
  }
  float sum = hsum(vsum);
  if (full < nk) {
    alignas(32) float tail[8] = {0};
    for (int j = full; j < nk; ++j) tail[j - full] = p[j] - maxv;
    alignas(32) float e[8];
    _mm256_store_ps(e, exp8(_mm256_load_ps(tail)));
    for (int j = full; j < nk; ++j) {
      p[j] = e[j - full];
      sum += e[j - full];
    }
  }
  return 1.0f / sum;
}

template <int HD>
void attentionT8(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk,
                 int heads, float *out, int ldo, float *p) {
  const __m256 scale = _mm256_set1_ps(1.0f / std::sqrt(float(HD)));
  const int chunks = (nk + 7) / 8;
  for (int i = 0; i < nq; ++i) {
    for (int h = 0; h < heads; ++h) {
      const float *qh = q + size_t(i) * ldq + h * HD;
      const float *kh = kt + size_t(h) * HD * ldkt;
      // scores, 4 chunks of 8 keys per pass over the head's dimensions
      for (int c0 = 0; c0 < chunks; c0 += 4) {
        const int nc = std::min(4, chunks - c0);
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int d = 0; d < HD; ++d) {
          const __m256 qd = _mm256_broadcast_ss(qh + d);
          const float *row = kh + size_t(d) * ldkt + c0 * 8;
          for (int u = 0; u < nc; ++u) acc[u] = _mm256_fmadd_ps(qd, _mm256_loadu_ps(row + u * 8), acc[u]);
        }
        for (int u = 0; u < nc; ++u) _mm256_storeu_ps(p + (c0 + u) * 8, _mm256_mul_ps(acc[u], scale));
      }
      const float inv = softmax8(p, nk);
      __m256 o[HD / 8];
      for (int c = 0; c < HD / 8; ++c) o[c] = _mm256_setzero_ps();
      for (int j = 0; j < nk; ++j) {
        const __m256 w = _mm256_set1_ps(p[j]);
        const float *vh = v + size_t(j) * ldv + h * HD;
        for (int c = 0; c < HD / 8; ++c) o[c] = _mm256_fmadd_ps(w, _mm256_loadu_ps(vh + c * 8), o[c]);
      }
      float *oh = out + size_t(i) * ldo + h * HD;
      const __m256 vinv = _mm256_set1_ps(inv);
      for (int c = 0; c < HD / 8; ++c) _mm256_storeu_ps(oh + c * 8, _mm256_mul_ps(o[c], vinv));
    }
  }
}

}  // namespace

void attentionT(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk,
                int heads, int headDim, float *out, int ldo, float *scratch) {
  if (useAvx512()) return attentionT512(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, headDim, out, ldo, scratch);
  switch (headDim) {
    case 32: return attentionT8<32>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    case 48: return attentionT8<48>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    case 64: return attentionT8<64>(q, ldq, nq, kt, ldkt, v, ldv, nk, heads, out, ldo, scratch);
    default: break;
  }
  // Any other head width: plain scalar code (no Firefox model uses one).
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

void ssru(const float *xwf, float *const *cells, int M, int dim, const LayerNorm &norm, float *x) {
  const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
  for (int m = 0; m < M; ++m) {
    const float *xw = xwf + size_t(m) * 2 * dim, *xf = xw + dim;
    float *c = cells[m], *row = x + size_t(m) * dim;
    for (int i = 0; i < dim; i += 8) {
      const __m256 f = _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_sub_ps(zero, _mm256_loadu_ps(xf + i)))));
      const __m256 cn = _mm256_fmadd_ps(f, _mm256_loadu_ps(c + i), _mm256_mul_ps(_mm256_sub_ps(one, f), _mm256_loadu_ps(xw + i)));
      _mm256_storeu_ps(c + i, cn);
      // residual: relu(c) + x, normalized below
      _mm256_storeu_ps(row + i, _mm256_add_ps(_mm256_max_ps(cn, zero), _mm256_loadu_ps(row + i)));
    }
  }
  addLayerNorm(x, nullptr, M, dim, norm, x);
}

}  // namespace reseam::mt::ops
