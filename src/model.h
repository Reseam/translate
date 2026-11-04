// A Firefox Translations model (Marian transformer encoder + SSRU decoder, int8 "alphas"),
// repacked for our kernels.
//
// All weights live in one storage block laid out exactly as the kernels read them. The first
// load of a model builds it from Marian's binary format and saves it as a cache file next to the
// model; later loads just memory-map that file. Mapped pages are clean and file-backed: loading
// is instant, processes share one copy, and under memory pressure the kernel can drop the pages
// of a model nobody is using and read them back later, instead of holding them in swap.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gemm.h"

namespace reseam::mt {

struct LayerNorm {
  const float *gamma = nullptr, *beta = nullptr;
};

struct EncoderLayer {
  PackedB qkv;   // self-attention Q|K|V fused (they share one activation scale)
  PackedB out;   // attention output projection Wo
  LayerNorm attnNorm;
  PackedB ffn1;  // ReLU, requantized straight into ffn2's input
  PackedB ffn2;
  LayerNorm ffnNorm;
};

struct DecoderLayer {
  PackedB rnn;   // SSRU W|Wf fused: [x * W | x * Wf + bf]
  LayerNorm rnnNorm;
  PackedB ctxQ;  // cross-attention query
  PackedB ctxKV; // cross-attention K|V, applied once to the encoder output
  PackedB ctxOut;
  LayerNorm ctxNorm;
  PackedB ffn1, ffn2;
  LayerNorm ffnNorm;
};

/// int8 embedding rows, dequantized (and scaled by sqrt(dim)) on lookup.
struct Embedding {
  const int8_t *rows = nullptr;  // vocab x dim
  float scale = 0;               // sqrt(dim) / quantMultB
  int vocab = 0, dim = 0;
  void lookup(int32_t id, float *out) const;
};

struct Model {
  int dim = 0, heads = 0, ffnDim = 0, encDepth = 0, decDepth = 0;
  Embedding srcEmbed, trgEmbed;  // the same rows twice when the model ties all embeddings
  // Output layer: logits = h * trgEmbed^T + b, scored per candidate word straight from the
  // target embedding rows (see argmaxCandidates), so no separate output matrix is kept.
  float outQuantMultA = 0;          // activation scale before the output layer
  float outScale = 0;               // 1 / (outQuantMultA * embedding quant mult)
  const float *outBias = nullptr;   // per word: bias - 127 * rowsum(word) * outScale
  std::vector<EncoderLayer> encoder;
  std::vector<DecoderLayer> decoder;
  const float *positions = nullptr; // sinusoidal position embeddings, kMaxPositions x dim
                                    // (one table per width, shared by every model)

  std::shared_ptr<const void> storage;  // keeps the weights alive (heap block or file mapping)
  size_t storageBytes = 0;
  bool mapped = false;  // storage is a memory-mapped cache file

  static constexpr int kMaxPositions = 1024;

  /// Asks the kernel to drop the weights' pages now (mapped storage only): for a model nobody has
  /// used for a while. They are read back from the cache file on the next use.
  void pageOut() const;

  /// Loads a Marian binary model (*.intgemm.alphas.bin). With `cachePath`, maps the packed cache
  /// file if it matches the model, and otherwise builds it and writes it there (falling back to
  /// keeping it in memory if the file cannot be written). Throws on anything unsupported.
  static Model load(const std::string &path, const std::string &cachePath = "");
};

}  // namespace reseam::mt
