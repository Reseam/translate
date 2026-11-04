#include "batch.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "ops.h"

namespace reseam::mt {

namespace fs = std::filesystem;

namespace {

constexpr float kMaxLengthFactor = 2.0f;  // Marian's max-length-factor: output tokens <= 2 x source
constexpr int kColumnAlign = 32;          // column slices per thread: whole AMX tile pairs

// The file in `dir` named <prefix>...<suffix>. Suffixes matter: our own model.reseam cache (and
// its temporary files) also start with "model.".
fs::path findFile(const fs::path &dir, const std::string &prefix, const std::string &suffix) {
  for (const auto &e : fs::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (e.is_regular_file() && name.size() > prefix.size() + suffix.size() && name.rfind(prefix, 0) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return e.path();
  }
  return {};
}

// Prints rows like Marian's debug() (first and last 5 columns) when RESEAM_MT_DEBUG is set, to
// compare intermediate values against an instrumented Marian build. Single-thread runs only.
const bool kDebug = std::getenv("RESEAM_MT_DEBUG") != nullptr;
void dump(const char *name, int index, const float *x, int rows, int cols) {
  if (!kDebug) return;
  std::fprintf(stderr, "Debug: %s%s\n", name, index >= 0 ? std::to_string(index).c_str() : "");
  for (int r = 0; r < rows; ++r) {
    const float *row = x + size_t(r) * cols;
    std::fprintf(stderr, "  [");
    for (int c = 0; c < 5; ++c) std::fprintf(stderr, " %13.8f", row[c]);
    std::fprintf(stderr, " ...");
    for (int c = cols - 5; c < cols; ++c) std::fprintf(stderr, " %13.8f", row[c]);
    std::fprintf(stderr, " ]\n");
  }
}

Epilogue store(float *out, int ld) {
  Epilogue e;
  e.kind = Epilogue::kStoreFloat;
  e.out = out;
  e.ld = ld;
  return e;
}

Epilogue reluQuant(uint8_t *out, int ld, float nextQuantMultA) {
  Epilogue e;
  e.kind = Epilogue::kReluQuant;
  e.outU8 = out;
  e.ld = ld;
  e.nextQuantMultA = nextQuantMultA;
  return e;
}

// Cycle accounting per operation kind, per thread (RESEAM_MT_PROFILE).
enum Op { kOpGemmEnc, kOpGemmDec, kOpGemmOut, kOpAttnEnc, kOpAttnDec, kOpNorm, kOpQuant, kOpEmbed, kOpSsru, kOpCount };
const char *kOpNames[] = {"gemm-enc", "gemm-dec", "gemm-out", "attn-enc", "attn-dec", "norm", "quant", "embed", "ssru"};
const bool kProfile = std::getenv("RESEAM_MT_PROFILE") != nullptr;
std::atomic<uint64_t> gCycles[kOpCount];
struct Timer {
  Op op;
  uint64_t t0;
  explicit Timer(Op o) : op(o), t0(kProfile ? __rdtsc() : 0) {}
  ~Timer() {
    if (kProfile) gCycles[op].fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
  }
};
struct Report {
  ~Report() {
    if (!kProfile) return;
    uint64_t total = 0;
    for (auto &c : gCycles) total += c.load();
    std::fprintf(stderr, "op cycles (all threads):");
    for (int i = 0; i < kOpCount; ++i) std::fprintf(stderr, " %s %.1f%%", kOpNames[i], 100.0 * gCycles[i].load() / (total ? total : 1));
    std::fprintf(stderr, "  total %.2f Gcycles\n", total / 1e9);
  }
} gReport;
thread_local Op tGemmOp = kOpGemmEnc;

}  // namespace

void Split::mm(const uint8_t *A, int lda, const PackedB &B, const Epilogue &ep) const {
  if (columns) {
    team.barrier();  // A is complete
    int c0, c1;
    Team::split(B.N, t, n, kColumnAlign, c0, c1);
    {
      Timer timer(tGemmOp);
      gemm(A, lda, M, B, c0, c1, ep);
    }
    team.barrier();  // the product is complete
  } else if (r1 > r0) {
    Timer timer(tGemmOp);
    gemm(A + size_t(r0) * lda, lda, r1 - r0, B, 0, B.N, ep.fromRow(r0));
  }
}

// ---- Bundle -------------------------------------------------------------------------------

Bundle::Bundle(const std::string &dirName, const std::string &cachePath) {
  const fs::path dir(dirName);
  const fs::path modelFile = findFile(dir, "model.", ".bin"), lexFile = findFile(dir, "lex.", ".bin");
  if (modelFile.empty() || lexFile.empty()) throw std::runtime_error("no model.*.bin or lex.*.bin in " + dirName);
  model = Model::load(modelFile.string(), cachePath);
  shortlist = std::make_unique<Shortlist>(lexFile.string());
  const fs::path shared = findFile(dir, "vocab.", ".spm");
  if (!shared.empty()) {
    srcVocab = trgVocab = Vocab::shared(shared.string());
    sharedVocab = true;
  } else {
    const fs::path src = findFile(dir, "srcvocab.", ".spm"), trg = findFile(dir, "trgvocab.", ".spm");
    if (src.empty() || trg.empty()) throw std::runtime_error("no vocabulary in " + dirName);
    srcVocab = Vocab::shared(src.string());
    trgVocab = Vocab::shared(trg.string());
  }
}

std::vector<int32_t> Bundle::candidates(const std::vector<int32_t> &src) const {
  std::vector<int32_t> ids = shortlist->candidates(src.data(), int(src.size()), sharedVocab, model.trgEmbed.vocab);
  const int32_t unk = trgVocab->unk();
  ids.erase(std::remove(ids.begin(), ids.end(), unk), ids.end());
  return ids;
}

// ---- Batch: admission (encoder) -----------------------------------------------------------

Batch::Batch(const Bundle &bundle, int threads)
    : b_(bundle), m_(bundle.model), threads_(threads), attnScratch_(threads), encScratch_(threads),
      keyScratch_(threads) {}

void Batch::admitBuffers(Scratch &scratch) {
  const size_t T = size_t(encRows_), dim = size_t(m_.dim);
  ex_ = scratch.take<float>(T * dim);
  ewide_ = scratch.take<float>(T * 3 * dim);
  ectx_ = scratch.take<float>(T * dim);
  etmp_ = scratch.take<float>(T * dim);
  exq_ = scratch.take<uint8_t>(T * dim);
  ehq_ = scratch.take<uint8_t>(T * m_.ffnDim);
  const size_t stride = ops::keyStride(longestAdmitting_);
  for (int t = 0; t < threads_; ++t) {
    encScratch_[t] = scratch.take<float>(stride);
    keyScratch_[t] = scratch.take<float>(dim * stride);
  }
}

void Batch::stepBuffers(Scratch &scratch) {
  const size_t M = size_t(stepRows_), dim = size_t(m_.dim);
  x_ = scratch.take<float>(M * dim);
  wide_ = scratch.take<float>(M * 2 * dim);
  ctx_ = scratch.take<float>(M * dim);
  tmp_ = scratch.take<float>(M * dim);
  att_ = scratch.take<float>(M * dim);
  xq_ = scratch.take<uint8_t>(M * dim);
  hq_ = scratch.take<uint8_t>(M * m_.ffnDim);
  for (int t = 0; t < threads_; ++t) attnScratch_[t] = scratch.take<float>(ops::keyStride(longestActive_));
}

int Batch::prepareAdmit(std::vector<Sentence *> sentences) {
  admitting_ = std::move(sentences);
  const int S = int(admitting_.size());
  int T = 0, longest = 0;
  for (auto *s : admitting_) {
    T += int(s->src.size());
    longest = std::max(longest, int(s->src.size()));
  }
  firstRow_.resize(S + 1);
  keyOffset_.resize(S + 1);
  rowSentence_.resize(T);
  rowStart_.resize(T);
  rowLen_.resize(T);
  rowPos_.resize(T);
  rowIds_.resize(T);
  const int dim = m_.dim;
  keyOffset_[0] = 0;
  for (int i = 0, row = 0; i < S; ++i) {
    firstRow_[i] = row;
    const auto &src = admitting_[i]->src;
    keyOffset_[i + 1] = keyOffset_[i] + size_t(dim) * ops::keyStride(int(src.size()));
    for (int p = 0; p < int(src.size()); ++p, ++row) {
      rowSentence_[row] = i;
      rowStart_[row] = row - p;
      rowLen_[row] = int(src.size());
      rowPos_[row] = p;
      rowIds_[row] = src[p];
    }
  }
  firstRow_[S] = T;
  longestAdmitting_ = longest;
  enc_ = std::make_shared<Encoded>();
  enc_->tokens = T;
  enc_->keyFloats = keyOffset_[S];
  enc_->keys = alignedAlloc<float>(size_t(m_.decDepth) * enc_->keyFloats);
  enc_->values = alignedAlloc<float>(size_t(m_.decDepth) * T * dim);
  encRows_ = T;
  return T;
}

void Batch::encodeRows(const Split &sp) {
  const int dim = m_.dim, ffn = m_.ffnDim, T = encRows_;
  const int t = sp.t, r0 = sp.r0, r1 = sp.r1, rows = sp.rows();
  float *x = ex_, *qkv = ewide_, *ctx = ectx_, *tmp = etmp_;
  uint8_t *xq = exq_, *hq = ehq_;
  float *xr = x + size_t(r0) * dim, *tr = tmp + size_t(r0) * dim;
  uint8_t *xqr = xq + size_t(r0) * dim;
  tGemmOp = kOpGemmEnc;
  {
  Timer timer(kOpEmbed);
  for (int r = r0; r < r1; ++r) {
    float *row = x + size_t(r) * dim;
    m_.srcEmbed.lookup(rowIds_[r], row);
    const float *p = m_.positions + size_t(rowPos_[r]) * dim;
    for (int d = 0; d < dim; ++d) row[d] += p[d];
  }
  }
  if (t == 0) dump("ENC_EMB", -1, x, T, dim);
  int layerNo = 0;
  for (const EncoderLayer &L : m_.encoder) {
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, L.qkv.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.qkv, store(qkv, 3 * dim));
    // Self-attention for this thread's rows, each against its own sentence.
    {
      // Self-attention for this thread's rows, each against its own sentence's keys (transposed
      // here into per-thread scratch; whole sentences in row mode, so each is transposed once).
      Timer timer(kOpAttnEnc);
      for (int r = r0; r < r1;) {
        const int start = rowStart_[r], len = rowLen_[r];
        const int nq = std::min(r1, start + len) - r;
        const int ldkt = ops::keyStride(len);
        float *kt = keyScratch_[t];
        ops::transposeKeys(qkv + size_t(start) * 3 * dim + dim, 3 * dim, len, dim, kt, ldkt);
        ops::attentionT(qkv + size_t(r) * 3 * dim, 3 * dim, nq, kt, ldkt, qkv + size_t(start) * 3 * dim + 2 * dim,
                        3 * dim, len, m_.heads, dim / m_.heads, ctx + size_t(r) * dim, dim, encScratch_[t]);
        r += nq;
      }
    }
    { Timer timer(kOpQuant); quantizeRows(ctx + size_t(r0) * dim, dim, rows, dim, L.out.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.out, store(tmp, dim));
    { Timer timer(kOpNorm); ops::addLayerNorm(tr, xr, rows, dim, L.attnNorm, xr); }
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, L.ffn1.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.ffn1, reluQuant(hq, ffn, L.ffn2.quantMultA));
    sp.mm(hq, ffn, L.ffn2, store(tmp, dim));
    { Timer timer(kOpNorm); ops::addLayerNorm(tr, xr, rows, dim, L.ffnNorm, xr); }
    if (kDebug) {
      sp.sync();
      if (t == 0) dump("ENC_L", ++layerNo, x, T, dim);
    }
  }
  // Project the encoder output once for every decoder layer's cross-attention: keys go out
  // transposed per sentence, values row-major. In row mode this thread owns whole sentences;
  // in column mode the sentences are shared out after the product.
  int s0, s1;
  if (sp.columns) {
    Team::split(int(admitting_.size()), t, sp.n, 1, s0, s1);
  } else {
    s0 = rows ? rowSentence_[r0] : 0;
    s1 = rows ? rowSentence_[r1 - 1] + 1 : 0;
  }
  for (int l = 0; l < m_.decDepth; ++l) {
    const PackedB &kv = m_.decoder[l].ctxKV;
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, kv.quantMultA, xqr, dim); }
    // [T][K | V], keeping the 3 * dim row stride of the QKV buffer: in row mode other threads may
    // still be running encoder layers on their rows of it, and a different stride would overlap them.
    sp.mm(xq, dim, kv, store(qkv, 3 * dim));
    Timer timer(kOpAttnEnc);
    float *keys = enc_->keys.get() + size_t(l) * enc_->keyFloats;
    float *values = enc_->values.get() + size_t(l) * T * dim;
    for (int i = s0; i < s1; ++i) {
      const int start = firstRow_[i], len = firstRow_[i + 1] - start;
      ops::transposeKeys(qkv + size_t(start) * 3 * dim, 3 * dim, len, dim, keys + keyOffset_[i], ops::keyStride(len));
      for (int r = start; r < start + len; ++r)
        std::memcpy(values + size_t(r) * dim, qkv + size_t(r) * 3 * dim + dim, dim * sizeof(float));
    }
    sp.sync();  // the next layer's product overwrites qkv
  }
}

void Batch::finishAdmit(EngineCounters &counters) {
  counters.encodedTokens += encRows_;
  for (int i = 0; i < int(admitting_.size()); ++i) {
    Sentence *s = admitting_[i];
    Active a;
    a.sentence = s;
    a.enc = enc_;
    a.offset = firstRow_[i];
    a.len = int(s->src.size());
    a.keyOffset = keyOffset_[i];
    a.keyStride = ops::keyStride(a.len);
    a.maxLen = std::max(1, int(kMaxLengthFactor * a.len));
    a.candidates = std::move(s->candidates);
    a.cells.assign(size_t(m_.decDepth) * m_.dim, 0.0f);
    active_.push_back(std::move(a));
  }
  admitting_.clear();
  enc_.reset();
  encRows_ = 0;
}

void Batch::drain(std::vector<Sentence *> &out) {
  for (auto &a : active_) out.push_back(a.sentence);
  out.insert(out.end(), admitting_.begin(), admitting_.end());
  active_.clear();
  admitting_.clear();
  enc_.reset();
  encRows_ = stepRows_ = 0;
}

// ---- Batch: decoder step ------------------------------------------------------------------

int Batch::prepareStep() {
  const int M = int(active_.size()), dim = m_.dim;
  stepRows_ = M;
  longestActive_ = 0;
  for (auto &a : active_) longestActive_ = std::max(longestActive_, a.len);
  cellPtrs_.resize(size_t(M) * m_.decDepth);
  for (int r = 0; r < M; ++r)
    for (int l = 0; l < m_.decDepth; ++l) cellPtrs_[size_t(l) * M + r] = active_[r].cells.data() + size_t(l) * dim;
  best_.assign(size_t(threads_) * M, -INFINITY);
  bestWord_.assign(size_t(threads_) * M, -1);
  return M;
}

void Batch::stepRows(const Split &sp) {
  const int M = stepRows_, dim = m_.dim, ffn = m_.ffnDim;
  const int t = sp.t, r0 = sp.r0, r1 = sp.r1, rows = sp.rows();
  float *x = x_, *xwf = wide_, *q = ctx_, *tmp = tmp_, *att = att_;
  uint8_t *xq = xq_, *hq = hq_;
  float *xr = x + size_t(r0) * dim, *tr = tmp + size_t(r0) * dim;
  uint8_t *xqr = xq + size_t(r0) * dim;
  tGemmOp = kOpGemmDec;
  {
    Timer timer(kOpEmbed);
    for (int r = r0; r < r1; ++r) {
      const Active &a = active_[r];
      float *row = x + size_t(r) * dim;
      if (a.last < 0)
        std::fill(row, row + dim, 0.0f);  // first step: zero embedding
      else
        m_.trgEmbed.lookup(a.last, row);
      const float *p = m_.positions + size_t(std::min(a.step, Model::kMaxPositions - 1)) * dim;
      for (int d = 0; d < dim; ++d) row[d] += p[d];
    }
  }
  if (t == 0) dump("DEC_EMB", -1, x, M, dim);
  for (int l = 0; l < m_.decDepth; ++l) {
    const DecoderLayer &L = m_.decoder[l];
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, L.rnn.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.rnn, store(xwf, 2 * dim));
    {
      Timer timer(kOpSsru);
      ops::ssru(xwf + size_t(r0) * 2 * dim, cellPtrs_.data() + size_t(l) * M + r0, rows, dim, L.rnnNorm, xr);
    }
    if (t == 0 && kDebug) dump("DEC_RNN", l + 1, x, M, dim);
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, L.ctxQ.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.ctxQ, store(q, dim));
    {
      Timer timer(kOpAttnDec);
      for (int r = r0; r < r1; ++r) {
        const Active &a = active_[r];
        const Encoded &e = *a.enc;
        ops::attentionT(q + size_t(r) * dim, dim, 1, e.keys.get() + size_t(l) * e.keyFloats + a.keyOffset, a.keyStride,
                        e.values.get() + (size_t(l) * e.tokens + a.offset) * dim, dim, a.len, m_.heads,
                        dim / m_.heads, att + size_t(r) * dim, dim, attnScratch_[t]);
      }
    }
    { Timer timer(kOpQuant); quantizeRows(att + size_t(r0) * dim, dim, rows, dim, L.ctxOut.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.ctxOut, store(tmp, dim));
    { Timer timer(kOpNorm); ops::addLayerNorm(tr, xr, rows, dim, L.ctxNorm, xr); }
    { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, L.ffn1.quantMultA, xqr, dim); }
    sp.mm(xq, dim, L.ffn1, reluQuant(hq, ffn, L.ffn2.quantMultA));
    sp.mm(hq, ffn, L.ffn2, store(tmp, dim));
    { Timer timer(kOpNorm); ops::addLayerNorm(tr, xr, rows, dim, L.ffnNorm, xr); }
    if (t == 0 && kDebug) dump("DEC_L", l + 1, x, M, dim);
  }
  // Output layer: each row scores only its own shortlist, straight from the int8 embedding rows.
  // Column mode splits every row's candidates across threads; finishStep merges the winners.
  { Timer timer(kOpQuant); quantizeRows(xr, dim, rows, dim, m_.outQuantMultA, xqr, dim); }
  Timer outTimer(kOpGemmOut);
  const int8_t *embed = m_.trgEmbed.rows;
  if (sp.columns) sp.team.barrier();
  const int o0 = sp.columns ? 0 : r0, o1 = sp.columns ? M : r1;
  for (int r = o0; r < o1; ++r) {
    const auto &cands = active_[r].candidates;
    int c0 = 0, c1 = int(cands.size());
    if (sp.columns) Team::split(int(cands.size()), t, sp.n, 16, c0, c1);
    argmaxCandidates(xq + size_t(r) * dim, embed, dim, cands.data() + c0, c1 - c0, m_.outScale, m_.outBias,
                     best_[size_t(t) * M + r], bestWord_[size_t(t) * M + r]);
  }
}

void Batch::finishStep(std::vector<Active> &done, EngineCounters &counters) {
  const int M = stepRows_;
  ++counters.decodeSteps;
  counters.decodeRows += M;
  // Merge per-thread winners; a later thread only wins with a strictly higher score, which is
  // the same tie-break as one scan through the candidates in order.
  const int32_t eos = b_.trgVocab->eos();
  for (int r = 0; r < M; ++r) {
    float best = -INFINITY;
    int32_t word = eos;
    for (int t = 0; t < threads_; ++t)
      if (best_[size_t(t) * M + r] > best) {
        best = best_[size_t(t) * M + r];
        word = bestWord_[size_t(t) * M + r];
      }
    bestWord_[r] = word;
  }
  size_t keep = 0;
  for (int r = 0; r < M; ++r) {
    Active &a = active_[r];
    const int32_t token = bestWord_[r];
    ++a.step;
    a.last = token;
    if (token != eos) {
      a.out.push_back(token);
      ++counters.tokens;
    }
    if (token == eos || a.step >= a.maxLen) {
      done.push_back(std::move(a));
    } else {
      if (keep != size_t(r)) active_[keep] = std::move(a);
      ++keep;
    }
  }
  active_.resize(keep);
  stepRows_ = 0;
}

}  // namespace reseam::mt
