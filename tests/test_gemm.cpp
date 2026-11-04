// Checks every available kernel against the scalar reference on awkward shapes and all epilogues.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gemm.h"

using namespace reseam::mt;

static int failures = 0;

static void check(bool ok, const char *what, Isa isa, int M, int K, int N) {
  if (!ok) {
    std::printf("FAIL %s isa=%s M=%d K=%d N=%d\n", what, isaName(isa), M, K, N);
    ++failures;
  }
}

int main() {
  std::mt19937 rng(42);
  // Activations as the model produces them: shifted values mostly within +-60 of 127. Wider
  // ranges would hit the documented AVX2 int16 saturation, which the reference does not model.
  // Weights stay within +-80 so AVX2's saturating int16 pair sums (max 187 * 80 * 2 < 32767)
  // cannot clip, which makes all paths exactly comparable. Real weights are checked separately.
  std::uniform_int_distribution<int> aDist(127 - 60, 127 + 60), bDist(-80, 80);
  std::uniform_real_distribution<float> fDist(-1.0f, 1.0f);
  std::vector<Isa> isas;
  for (Isa isa : {Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512Vnni, Isa::kAmx})
    if (forceIsa(isa)) isas.push_back(isa);
  std::printf("testing:");
  for (Isa isa : isas) std::printf(" %s", isaName(isa));
  std::printf("\n");

  for (int K : {64, 384, 1536})
    for (int N : {16, 48, 64, 80, 384, 1152})
      for (int M : {1, 2, 5, 6, 7, 13, 16, 17, 31, 32, 33, 64}) {
        std::vector<int8_t> raw(size_t(K) * N);
        for (auto &v : raw) v = int8_t(bDist(rng));
        std::vector<float> bias(N);
        for (auto &v : bias) v = fDist(rng);
        PackedOwner owner(K, N);
        owner.pack(raw.data(), K, N, N, 1, 0);
        owner.dequant(0, N, 11.0f, 120.0f, bias.data());
        const PackedB &B = owner.view;
        const int lda = B.K;
        std::vector<uint8_t> A(size_t(M) * lda);
        for (auto &v : A) v = uint8_t(aDist(rng));

        auto run = [&](Isa isa, std::vector<float> &f, std::vector<uint8_t> &q) {
          forceIsa(isa);
          f.assign(size_t(M) * N, 0);
          q.assign(size_t(M) * N, 0);
          Epilogue e;
          e.kind = Epilogue::kStoreFloat;
          e.out = f.data();
          e.ld = N;
          gemm(A.data(), lda, M, B, e);
          e = {};
          e.kind = Epilogue::kReluQuant;
          e.outU8 = q.data();
          e.ld = N;
          e.nextQuantMultA = 0.05f;
          gemm(A.data(), lda, M, B, e);
        };
        std::vector<float> rf;
        std::vector<uint8_t> rq;
        run(Isa::kReference, rf, rq);
        for (Isa isa : isas) {
          std::vector<float> f;
          std::vector<uint8_t> q;
          run(isa, f, q);
          bool okF = true, okQ = true;
          for (size_t i = 0; i < f.size(); ++i) {
            okF &= std::fabs(f[i] - rf[i]) <= 1e-4f * (1 + std::fabs(rf[i]));
            okQ &= std::abs(int(q[i]) - int(rq[i])) <= 1;  // rounding at exact .5 may differ by FMA
          }
          check(okF, "float", isa, M, K, N);
          check(okQ, "reluquant", isa, M, K, N);
        }
      }
  // Candidate scoring (output layer): random candidate lists of awkward lengths over a vocabulary.
  for (int K : {256, 384, 512}) {
    const int vocab = 3000;
    std::vector<int8_t> rows(size_t(vocab) * K);
    for (auto &v : rows) v = int8_t(bDist(rng));
    std::vector<float> bias(vocab);
    for (auto &v : bias) v = fDist(rng) * 50;
    std::vector<uint8_t> x(K);
    for (int n : {1, 7, 8, 15, 16, 17, 100, 1177}) {
      for (auto &v : x) v = uint8_t(aDist(rng));
      std::vector<int32_t> ids;
      for (int i = 0; i < n; ++i) ids.push_back(int32_t(rng() % vocab));
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      forceIsa(Isa::kReference);
      float rs = -INFINITY;
      int32_t rw = -1;
      argmaxCandidates(x.data(), rows.data(), K, ids.data(), int(ids.size()), 0.001f, bias.data(), rs, rw);
      for (Isa isa : isas) {
        forceIsa(isa);
        float s2 = -INFINITY;
        int32_t w2 = -1;
        argmaxCandidates(x.data(), rows.data(), K, ids.data(), int(ids.size()), 0.001f, bias.data(), s2, w2);
        check(w2 == rw && std::fabs(s2 - rs) <= 1e-4f * (1 + std::fabs(rs)), "candidates", isa, 1, K, n);
      }
    }
  }
  std::printf(failures ? "%d failures\n" : "all kernels match the reference\n", failures);
  return failures ? 1 : 0;
}
