// Float kernels around the int8 products: layer norm, attention, the SSRU update.
// AVX2 + FMA (the baseline every build targets).
#pragma once

#include <cstdint>

#include "model.h"

namespace reseam::mt::ops {

/// out[m] = LayerNorm(a[m] + b[m]) for M rows of `dim`, Marian's eps (1e-6). `out` may alias a or b.
void addLayerNorm(const float *a, const float *b, int M, int dim, const LayerNorm &norm, float *out);

/// Reference attention on row-major keys (tests check attentionT against it): `nq` query rows
/// against `nk` key/value rows, all heads. Row strides ldq/ldk/ldv; out: nq x dim.
void attention(const float *q, int ldq, int nq, const float *k, int ldk, const float *v, int ldv, int nk, int heads,
               int headDim, float *out, int ldo, float *scratch /* nk floats */);

/// Keys laid out for attentionT: kt[d * ldkt + j] = component d of key j, for `dim` components.
/// ldkt >= nk is rounded up to 16; the padding columns are zeroed.
void transposeKeys(const float *k, int ldk, int nk, int dim, float *kt, int ldkt);
inline int keyStride(int nk) { return (nk + 15) / 16 * 16; }

/// Attention against transposed keys (see transposeKeys): scores for 8 or 16 keys come out of one
/// vector FMA per head dimension, with no horizontal sums. v is row-major (stride ldv).
/// scratch: ldkt floats.
void attentionT(const float *q, int ldq, int nq, const float *kt, int ldkt, const float *v, int ldv, int nk,
                int heads, int headDim, float *out, int ldo, float *scratch);

/// SSRU step for M rows: f = sigmoid(xf), c = f * c + (1 - f) * xw, then
/// x = LayerNorm(relu(c) + x). xwf holds [xw | xf] per row (stride 2 * dim).
void ssru(const float *xwf, float *const *cells, int M, int dim, const LayerNorm &norm, float *x);

}  // namespace reseam::mt::ops
