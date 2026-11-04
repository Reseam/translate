// AMX kernel (Sapphire Rapids and later): one TDPBUSD multiplies a 16 x 64 u8 tile by a 64 x 16 s8
// tile into 16 x 16 int32, 16K multiply-adds per instruction. The packed B layout is already
// AMX's B-tile layout ([k/4][16 columns][4 k]), so tiles load straight from the weights.
//
// Register plan: 2 x 2 output tiles (32 rows x 32 columns) + 2 A tiles + 2 B tiles = 8 tiles.
// Rows that do not fill a 16-row tile go to the AVX-512 VNNI kernel, which is faster there.
// Compiled with -mamx-tile -mamx-int8 -mavx512f -mavx512bw -mavx512vl -mavx512vnni.
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "epilogue.h"

namespace reseam::mt::kernels {
namespace {

struct alignas(64) TileConfig {
  uint8_t palette = 1;
  uint8_t startRow = 0;
  uint8_t reserved[14] = {};
  uint16_t colsb[16] = {};
  uint8_t rows[16] = {};
};

void configureTiles() {
  thread_local bool configured = false;
  if (configured) return;
  TileConfig cfg;
  for (int t = 0; t < 8; ++t) {
    cfg.rows[t] = 16;
    cfg.colsb[t] = 64;  // C: 16 int32; A: 64 u8; B: 16 groups of 4 s8
  }
  _tile_loadconfig(&cfg);
  configured = true;
}

}  // namespace

bool amxEnable() {
  constexpr int kArchReqXcompPerm = 0x1023;
  constexpr int kXfeatureXtiledata = 18;
  return syscall(SYS_arch_prctl, kArchReqXcompPerm, kXfeatureXtiledata) == 0;
}

void gemmAmx(const uint8_t *A, int lda, int M, const PackedB &B, int n0, int n1, const Epilogue &ep) {
  const int mFull = M / 16 * 16;
  if (mFull == 0) return gemmAvx512Vnni(A, lda, M, B, n0, n1, ep);
  configureTiles();
  alignas(64) int32_t acc[32 * 32];
  const int b0 = n0 / kBlockN, b1 = n1 / kBlockN;
  const int kTiles = B.K / 64;
  for (int nb = b0; nb < b1; nb += 2) {
    const bool two = nb + 1 < b1;
    for (int m = 0; m < mFull; m += 32) {
      const bool tall = m + 16 < mFull;
      _tile_zero(0);
      _tile_zero(1);
      _tile_zero(2);
      _tile_zero(3);
      for (int kt = 0; kt < kTiles; ++kt) {
        const uint8_t *a = A + size_t(m) * lda + kt * 64;
        _tile_loadd(4, a, lda);
        if (tall) _tile_loadd(5, a + size_t(16) * lda, lda);
        _tile_loadd(6, B.block(nb) + size_t(kt) * 1024, 64);
        if (two) _tile_loadd(7, B.block(nb + 1) + size_t(kt) * 1024, 64);
        _tile_dpbusd(0, 4, 6);
        if (two) _tile_dpbusd(1, 4, 7);
        if (tall) {
          _tile_dpbusd(2, 5, 6);
          if (two) _tile_dpbusd(3, 5, 7);
        }
      }
      // acc is 32 rows x 32 columns; tiles land in their quadrant.
      _tile_stored(0, acc, 128);
      if (two) _tile_stored(1, acc + 16, 128);
      if (tall) {
        _tile_stored(2, acc + 16 * 32, 128);
        if (two) _tile_stored(3, acc + 16 * 32 + 16, 128);
      }
      applyEpilogue(acc, 32, m, tall ? 32 : 16, nb * kBlockN, two ? 32 : 16, B, ep);
    }
  }
  if (mFull < M) gemmAvx512Vnni(A + size_t(mFull) * lda, lda, M - mFull, B, n0, n1, ep.fromRow(mFull));
}

}  // namespace reseam::mt::kernels
