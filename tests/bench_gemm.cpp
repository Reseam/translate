// Kernel microbenchmark at the model's real shapes: multiply-adds per cycle for each ISA path.
// usage: bench_gemm [ghz=2.1]
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "gemm.h"

using namespace reseam::mt;

int main(int argc, char **argv) {
  const double ghz = argc > 1 ? std::stod(argv[1]) : 2.1;
  std::mt19937 rng(1);
  struct Shape {
    int M, K, N;
  };
  const std::vector<Shape> shapes = {{1, 384, 384},   {1, 384, 1536},   {1, 1536, 384},  {8, 384, 1536},
                                     {26, 384, 384},  {26, 384, 1536},  {26, 1536, 384}, {64, 384, 1536},
                                     {64, 1536, 384}, {650, 384, 1152}, {650, 384, 1536}, {650, 1536, 384}};
  std::printf("%-16s", "M x K x N");
  std::vector<Isa> isas;
  for (Isa isa : {Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512Vnni, Isa::kAmx})
    if (forceIsa(isa)) {
      isas.push_back(isa);
      std::printf("%14s", isaName(isa));
    }
  std::printf("   (multiply-adds per cycle per core)\n");
  for (const Shape &s : shapes) {
    PackedOwner owner(s.K, s.N);
    std::vector<int8_t> raw(size_t(s.K) * s.N);
    for (auto &v : raw) v = int8_t(int(rng() % 161) - 80);
    owner.pack(raw.data(), s.K, s.N, s.N, 1, 0);
    owner.dequant(0, s.N, 10.0f, 100.0f, nullptr);
    const PackedB &B = owner.view;
    std::vector<uint8_t> A(size_t(s.M) * B.K);
    for (auto &v : A) v = uint8_t(67 + rng() % 120);
    std::vector<float> C(size_t(s.M) * s.N);
    Epilogue ep;
    ep.out = C.data();
    ep.ld = s.N;
    std::printf("%4d x%5d x%5d", s.M, s.K, s.N);
    for (Isa isa : isas) {
      forceIsa(isa);
      const double macs = double(s.M) * s.K * s.N;
      const int reps = std::max(3, int(2e9 / macs / 20));
      gemm(A.data(), B.K, s.M, B, ep);  // warm
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) gemm(A.data(), B.K, s.M, B, ep);
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("%14.1f", macs * reps / secs / (ghz * 1e9));
    }
    std::printf("\n");
  }
}
