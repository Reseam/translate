// The compute side of the engine: every model's running batch, executed by one Team.
//
// A team job covers the rows of all models at once. How threads divide it depends on load:
//
//  - Column mode (few rows in total, the latency case): matrix products split by output columns,
//    so each thread only reads its own slice of every weight matrix (which stays in its L2) and a
//    single sentence gets every core. Per-row work splits by rows; barriers between phases.
//  - Row mode (many rows): the rows of all models are laid end to end and cut into one
//    contiguous range per thread; each thread takes its range through every layer on its own,
//    with no barriers. Small products are cheap enough that synchronizing cores around each one
//    costs more than it saves, so this wins once there is enough work to share.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gemm.h"
#include "model.h"
#include "team.h"
#include "text.h"

namespace reseam::mt {

/// A sentence as the batch sees it; the engine extends it with where the result goes.
struct Sentence {
  std::vector<int32_t> src;     // source ids, EOS last
  std::vector<int32_t> candidates;  // words it may produce, sorted (Bundle::candidates); set by the requester
  virtual ~Sentence() = default;
};

/// A model with what it needs around it: vocabularies and the lexical shortlist.
struct Bundle {
  Model model;
  std::shared_ptr<Vocab> srcVocab, trgVocab;
  bool sharedVocab = false;
  std::unique_ptr<Shortlist> shortlist;

  /// `cachePath`: where the packed weights are cached (see Model::load); empty for no cache.
  Bundle(const std::string &dir, const std::string &cachePath);
  /// The words a sentence may produce, sorted: its shortlist, minus <unk> (Marian never emits it).
  std::vector<int32_t> candidates(const std::vector<int32_t> &src) const;
};

struct EngineCounters {
  uint64_t sentences = 0, tokens = 0, decodeSteps = 0, decodeRows = 0, encodedTokens = 0, rowJobs = 0, columnJobs = 0;
};

/// Encoder output projected to every decoder layer's cross-attention keys and values, for the
/// sentences admitted together, shared by them until all finish. Keys are stored transposed per
/// sentence (ops::transposeKeys) so attention scores come 16 keys per instruction.
struct Encoded {
  AlignedPtr<float> keys;    // [decDepth][keyFloats]: per sentence, dim x keyStride(len)
  AlignedPtr<float> values;  // [decDepth][tokens][dim]
  size_t keyFloats = 0;      // per layer
  int tokens = 0;
};

struct Active {
  Sentence *sentence;
  std::shared_ptr<Encoded> enc;
  int offset, len;  // this sentence's rows in enc->values
  size_t keyOffset; // its transposed keys in each layer of enc->keys
  int keyStride;    // and their row stride
  int step = 0, maxLen;
  int32_t last = -1;
  std::vector<int32_t> out;
  std::vector<int32_t> candidates;  // shortlist: words this sentence may produce, sorted
  std::vector<float> cells;         // SSRU state, decDepth x dim
};

/// Working memory for one team job, shared by every model: activations, quantized inputs and
/// per-thread attention scratch are only needed while the job runs, so one block sized for the
/// largest job replaces a high-water-mark buffer per model. Used in two passes with the same
/// calls: sizing (begin, then every take returns null and only counts), then commit and take
/// again for real pointers.
class Scratch {
 public:
  void begin() {
    base_ = nullptr;
    used_ = 0;
  }
  void commit() {
    if (used_ > capacity_) {
      mem_ = alignedAlloc<char>(used_);
      capacity_ = used_;
    }
    base_ = mem_.get();
    used_ = 0;
  }
  template <class T>
  T *take(size_t count) {
    used_ = (used_ + 63) & ~size_t(63);
    T *p = base_ ? reinterpret_cast<T *>(base_ + used_) : nullptr;
    used_ += count * sizeof(T);
    return p;
  }
  /// Gives the memory back (when the engine goes idle).
  void release() {
    mem_.reset();
    capacity_ = 0;
  }
  size_t capacity() const { return capacity_; }

 private:
  AlignedPtr<char> mem_;
  char *base_ = nullptr;
  size_t capacity_ = 0, used_ = 0;
};

/// Which rows of a phase this thread works on, and whether products split by columns instead.
struct Split {
  Team &team;
  int t, n;      // this thread, team size
  bool columns;  // column mode: products split by columns, rows [r0, r1) only for per-row work
  int M;         // rows of the batch being processed
  int r0, r1;    // this thread's rows

  int rows() const { return r1 - r0; }
  /// A product over all M rows (column mode) or this thread's rows (row mode).
  void mm(const uint8_t *A, int lda, const PackedB &B, const Epilogue &ep) const;
  void sync() const {
    if (columns) team.barrier();
  }
};

class Batch {
 public:
  Batch(const Bundle &bundle, int threads);

  const Bundle &bundle() const { return b_; }
  bool empty() const { return active_.empty(); }
  int size() const { return int(active_.size()); }

  // Admitting sentences: prepareAdmit and admitBuffers (serial) -> encodeRows (every thread) ->
  // finishAdmit (serial). In row mode a thread's range must hold whole sentences;
  // sentenceRows() gives the cut points.
  /// Returns the number of encoder rows (source tokens).
  int prepareAdmit(std::vector<Sentence *> sentences);
  /// Takes the encoder's working memory from `scratch` (see Scratch for the two passes).
  void admitBuffers(Scratch &scratch);
  const std::vector<int> &sentenceRows() const { return firstRow_; }  // S + 1 cut points
  void encodeRows(const Split &sp);
  void finishAdmit(EngineCounters &counters);

  /// Removes every sentence (active or being admitted) and appends it to `out`; the batch is
  /// left empty. For failing them back after an error.
  void drain(std::vector<Sentence *> &out);

  // A decoder step: prepareStep (serial) -> stepRows (every thread) -> finishStep (serial).
  /// Returns the number of rows (active sentences).
  int prepareStep();
  /// Takes the decoder step's working memory from `scratch`.
  void stepBuffers(Scratch &scratch);
  void stepRows(const Split &sp);
  void finishStep(std::vector<Active> &done, EngineCounters &counters);

 private:
  const Bundle &b_;
  const Model &m_;
  int threads_;
  std::vector<Active> active_;
  // Working memory for the current job, from the shared Scratch. Decoder and encoder have
  // separate pieces: one team job can run both at once.
  float *x_ = nullptr, *wide_ = nullptr, *ctx_ = nullptr, *tmp_ = nullptr, *att_ = nullptr;
  uint8_t *xq_ = nullptr, *hq_ = nullptr;
  float *ex_ = nullptr, *ewide_ = nullptr, *ectx_ = nullptr, *etmp_ = nullptr;
  uint8_t *exq_ = nullptr, *ehq_ = nullptr;
  std::vector<float *> attnScratch_, encScratch_, keyScratch_;  // per thread
  int longestActive_ = 0, longestAdmitting_ = 0;
  std::vector<float *> cellPtrs_;
  std::vector<float> best_;       // threads x rows: best score so far
  std::vector<int32_t> bestWord_; // threads x rows: its word

  // Admission in progress.
  std::vector<Sentence *> admitting_;
  std::vector<int> firstRow_, rowStart_, rowLen_, rowPos_, rowSentence_;
  std::vector<size_t> keyOffset_;
  std::vector<int32_t> rowIds_;
  std::shared_ptr<Encoded> enc_;
  int encRows_ = 0;
  // Step in progress.
  int stepRows_ = 0;
};

}  // namespace reseam::mt
