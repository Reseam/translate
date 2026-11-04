#include "model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace reseam::mt {

namespace {

constexpr uint64_t kTypeFloat32 = 0x404;
constexpr uint64_t kTypeIntgemm8 = 0x4101;

struct Item {
  uint64_t type = 0;
  std::vector<int> shape;
  const char *data = nullptr;
  size_t bytes = 0;
};

struct File {
  std::vector<char> buffer;
  std::map<std::string, Item> items;
  std::map<std::string, std::string> config;  // flat keys from the embedded model.yml

  const Item &get(const std::string &name) const {
    auto it = items.find(name);
    if (it == items.end()) throw std::runtime_error("model has no " + name);
    return it->second;
  }
  bool has(const std::string &name) const { return items.count(name) > 0; }
  int intConfig(const std::string &key) const {
    auto it = config.find(key);
    if (it == config.end()) throw std::runtime_error("model.yml has no " + key);
    return std::stoi(it->second);
  }
  std::string strConfig(const std::string &key) const {
    auto it = config.find(key);
    return it == config.end() ? "" : it->second;
  }
  std::vector<float> floats(const std::string &name) const {
    const Item &it = get(name);
    if (it.type != kTypeFloat32) throw std::runtime_error(name + " is not float32");
    std::vector<float> v(it.bytes / 4);
    std::memcpy(v.data(), it.data, v.size() * 4);
    return v;
  }
  float scalar(const std::string &name) const { return floats(name).at(0); }
};

// The whole file in one read (istreambuf_iterator goes byte by byte: seconds for a 60 MB model).
std::vector<char> readFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<char> data(size_t(in.tellg()));
  in.seekg(0);
  if (!in.read(data.data(), std::streamsize(data.size()))) throw std::runtime_error("cannot read " + path);
  return data;
}

template <class T>
T read(const char *&p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  p += sizeof(T);
  return v;
}

File parse(const std::string &path) {
  File f;
  f.buffer = readFile(path);
  const char *p = f.buffer.data(), *end = p + f.buffer.size();
  if (f.buffer.size() < 16 || read<uint64_t>(p) != 1) throw std::runtime_error(path + ": not a Marian binary model");
  const uint64_t n = read<uint64_t>(p);
  struct Header { uint64_t nameLength, type, shapeLength, dataLength; };
  std::vector<Header> headers(n);
  for (auto &h : headers) h = read<Header>(p);
  std::vector<std::string> names(n);
  for (uint64_t i = 0; i < n; ++i) {
    names[i].assign(p, headers[i].nameLength ? headers[i].nameLength - 1 : 0);
    p += headers[i].nameLength;
  }
  std::vector<std::vector<int>> shapes(n);
  for (uint64_t i = 0; i < n; ++i)
    for (uint64_t k = 0; k < headers[i].shapeLength; ++k) shapes[i].push_back(read<int32_t>(p));
  p += read<uint64_t>(p);  // padding to the data section
  for (uint64_t i = 0; i < n; ++i) {
    if (p + headers[i].dataLength > end) throw std::runtime_error(path + ": truncated");
    f.items[names[i]] = {headers[i].type, shapes[i], p, headers[i].dataLength};
    p += headers[i].dataLength;
  }
  // model.yml: only the flat "key: value" lines matter here.
  if (f.has("special:model.yml")) {
    const Item &y = f.get("special:model.yml");
    std::istringstream ys(std::string(y.data, strnlen(y.data, y.bytes)));
    for (std::string line; std::getline(ys, line);) {
      auto colon = line.find(':');
      if (colon == std::string::npos || line[0] == ' ') continue;
      std::string value = line.substr(colon + 1);
      value.erase(0, value.find_first_not_of(' '));
      f.config[line.substr(0, colon)] = value;
    }
  }
  return f;
}

// An intgemm8 item: K x N int8 followed by its float quant mult. Weight matrices are stored
// transposed (N rows of K) while the header keeps the K x N shape; embeddings are plain rows.
struct Int8Matrix {
  const int8_t *values;
  int K, N;
  float quantMult;
};

Int8Matrix int8Matrix(const File &f, const std::string &name) {
  const Item &it = f.get(name);
  if (it.type != kTypeIntgemm8 || it.shape.size() != 2) throw std::runtime_error(name + " is not an int8 matrix");
  Int8Matrix m{reinterpret_cast<const int8_t *>(it.data), it.shape[0], it.shape[1], 0};
  std::memcpy(&m.quantMult, it.data + size_t(m.K) * m.N, sizeof(float));
  return m;
}

void require(const File &f, const std::string &key, const std::string &value) {
  if (f.strConfig(key) != value)
    throw std::runtime_error("unsupported model: " + key + " is '" + f.strConfig(key) + "', expected '" + value + "'");
}

// ---- Storage layout ---------------------------------------------------------------------------

constexpr char kMagic[8] = {'R', 'S', 'M', 'T', 'P', 'A', 'C', 'K'};
constexpr uint32_t kFormatVersion = 1;  // bump on any change to the layout below

// Start of the storage block (and of the cache file).
struct Header {
  char magic[8];
  uint32_t version, headerBytes;
  uint64_t totalBytes;
  uint64_t sourceSize;
  int64_t sourceMtime;
  int32_t dim, heads, ffnDim, encDepth, decDepth, srcVocab, trgVocab, tied;
  float srcEmbedScale, trgEmbedScale, outQuantMultA, outScale;
};

// Hands out 64-byte aligned ranges of the storage block in a fixed order. The same walk sizes the
// block (no base), fills it from a Marian model, or points into a mapped cache file, so the three
// can never disagree about the layout.
class Arena {
 public:
  explicit Arena(char *base) : base_(base) {}
  template <class T>
  T *take(size_t count) {
    offset_ = (offset_ + 63) & ~size_t(63);
    T *p = base_ ? reinterpret_cast<T *>(base_ + offset_) : nullptr;
    offset_ += count * sizeof(T);
    return p;
  }
  bool real() const { return base_ != nullptr; }
  size_t size() const { return offset_; }

 private:
  char *base_;
  size_t offset_ = 0;
};

template <class T>
T *mut(const T *p) {
  return const_cast<T *>(p);
}

// Walks the layout for `m` (whose dimensions are set). With `src`, fills the block from the Marian
// model as it goes; otherwise only points the model at what is already there.
class Layout {
 public:
  Layout(Model &m, Arena &arena, const File *src) : m_(m), a_(arena), src_(src) {}

  void run(bool tied, const std::string &srcName, const std::string &trgName) {
    embedding(m_.srcEmbed, srcName);
    if (tied)
      m_.trgEmbed = m_.srcEmbed;
    else
      embedding(m_.trgEmbed, trgName);
    m_.outBias = a_.take<float>(m_.trgEmbed.vocab);
    if (filling()) outputBias(trgName);
    m_.encoder.resize(m_.encDepth);
    for (int l = 0; l < m_.encDepth; ++l) {
      const std::string p = "encoder_l" + std::to_string(l + 1);
      EncoderLayer &L = m_.encoder[l];
      L.qkv = packed(m_.dim, 3 * m_.dim, {p + "_self_Wq", p + "_self_Wk", p + "_self_Wv"}, {p + "_self_bq", p + "_self_bk", p + "_self_bv"});
      L.out = packed(m_.dim, m_.dim, {p + "_self_Wo"}, {p + "_self_bo"});
      L.attnNorm = norm(p + "_self_Wo");
      L.ffn1 = packed(m_.dim, m_.ffnDim, {p + "_ffn_W1"}, {p + "_ffn_b1"});
      L.ffn2 = packed(m_.ffnDim, m_.dim, {p + "_ffn_W2"}, {p + "_ffn_b2"});
      L.ffnNorm = norm(p + "_ffn_ffn");
    }
    m_.decoder.resize(m_.decDepth);
    for (int l = 0; l < m_.decDepth; ++l) {
      const std::string p = "decoder_l" + std::to_string(l + 1);
      DecoderLayer &L = m_.decoder[l];
      L.rnn = packed(m_.dim, 2 * m_.dim, {p + "_rnn_W", p + "_rnn_Wf"}, {"", p + "_rnn_bf"});
      L.rnnNorm = norm(p + "_rnn_ffn");
      L.ctxQ = packed(m_.dim, m_.dim, {p + "_context_Wq"}, {p + "_context_bq"});
      L.ctxKV = packed(m_.dim, 2 * m_.dim, {p + "_context_Wk", p + "_context_Wv"}, {p + "_context_bk", p + "_context_bv"});
      L.ctxOut = packed(m_.dim, m_.dim, {p + "_context_Wo"}, {p + "_context_bo"});
      L.ctxNorm = norm(p + "_context_Wo");
      L.ffn1 = packed(m_.dim, m_.ffnDim, {p + "_ffn_W1"}, {p + "_ffn_b1"});
      L.ffn2 = packed(m_.ffnDim, m_.dim, {p + "_ffn_W2"}, {p + "_ffn_b2"});
      L.ffnNorm = norm(p + "_ffn_ffn");
    }
  }

 private:
  bool filling() const { return src_ && a_.real(); }

  void embedding(Embedding &e, const std::string &name) {
    e.dim = m_.dim;
    if (src_) {
      const Int8Matrix w = int8Matrix(*src_, name);
      e.vocab = w.K;
      e.scale = std::sqrt(float(w.N)) / w.quantMult;  // Marian scales input embeddings by sqrt(dim)
      if (w.N != m_.dim) throw std::runtime_error(name + " has the wrong width");
    }
    e.rows = a_.take<int8_t>(size_t(e.vocab) * e.dim);
    if (filling()) std::memcpy(mut(e.rows), int8Matrix(*src_, name).values, size_t(e.vocab) * e.dim);
  }

  void outputBias(const std::string &trgName) {
    const Int8Matrix e = int8Matrix(*src_, trgName);
    const auto bias = src_->floats("decoder_ff_logit_out_b");
    for (int w = 0; w < e.K; ++w) {
      int32_t rowsum = 0;
      for (int k = 0; k < e.N; ++k) rowsum += e.values[size_t(w) * e.N + k];
      // Activations are shifted by +127, so 127 * rowsum comes back out here.
      mut(m_.outBias)[w] = bias[w] - 127.0f * float(rowsum) * m_.outScale;
    }
  }

  // Matrices that share one input (and so one activation scale) packed side by side.
  PackedB packed(int K, int N, const std::vector<std::string> &names, const std::vector<std::string> &biases) {
    PackedB b;
    b.K = paddedK(K);
    b.N = paddedN(N);
    float *qa = a_.take<float>(1);
    b.data = a_.take<int8_t>(size_t(b.K) * b.N);
    b.scale = a_.take<float>(b.N);
    b.bias = a_.take<float>(b.N);
    if (!a_.real()) return b;
    if (!src_) {
      b.quantMultA = *qa;
      return b;
    }
    const float qA = src_->scalar(names[0] + "_QuantMultA");
    for (const auto &name : names)
      if (src_->scalar(name + "_QuantMultA") != qA) throw std::runtime_error("cannot fuse " + name + ": different input scale");
    *qa = b.quantMultA = qA;
    std::memset(mut(b.data), 0, size_t(b.K) * b.N);
    int col = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      const Int8Matrix w = int8Matrix(*src_, names[i]);
      if (w.K != K) throw std::runtime_error(names[i] + " has the wrong shape");
      // Stored transposed: element (k, n) at values[n * K + k].
      packColumns(w.values, w.K, w.N, 1, size_t(w.K), mut(b.data), b.K, col);
      const std::vector<float> bias = biases[i].empty() ? std::vector<float>() : src_->floats(biases[i]);
      dequantColumns(b.data, b.K, col, w.N, qA, w.quantMult, bias.empty() ? nullptr : bias.data(), mut(b.scale), mut(b.bias));
      col += w.N;
    }
    if (col != N) throw std::runtime_error(names[0] + ": fused width mismatch");
    return b;
  }

  LayerNorm norm(const std::string &prefix) {
    LayerNorm n{a_.take<float>(m_.dim), a_.take<float>(m_.dim)};
    if (filling()) {
      const auto g = src_->floats(prefix + "_ln_scale"), b = src_->floats(prefix + "_ln_bias");
      if (int(g.size()) != m_.dim || int(b.size()) != m_.dim) throw std::runtime_error(prefix + ": bad layer norm");
      std::memcpy(mut(n.gamma), g.data(), m_.dim * sizeof(float));
      std::memcpy(mut(n.beta), b.data(), m_.dim * sizeof(float));
    }
    return n;
  }

  Model &m_;
  Arena &a_;
  const File *src_;
};

// Marian's sinusoidal embeddings: sin in the first half, cos in the second. One table per
// width for the whole process (they never change).
const float *sinusoidalPositions(int dim) {
  static std::mutex mu;
  static std::map<int, std::vector<float>> tables;
  std::lock_guard<std::mutex> lock(mu);
  auto [it, fresh] = tables.try_emplace(dim);
  std::vector<float> &out = it->second;
  if (!fresh) return out.data();
  out.assign(size_t(Model::kMaxPositions) * dim, 0.0f);
  const float half = float(dim) / 2;
  const float increment = std::log(10000.f) / (half - 1.f);
  for (int p = 0; p < Model::kMaxPositions; ++p)
    for (int i = 0; i < int(half); ++i) {
      const float v = p * std::exp(i * -increment);
      out[size_t(p) * dim + i] = std::sin(v);
      out[size_t(p) * dim + int(half) + i] = std::cos(v);
    }
  return out.data();
}

// Points `m` into a mapped cache file if it is valid for this source model; false otherwise.
bool mapCache(Model &m, const std::string &cachePath, uint64_t sourceSize, int64_t sourceMtime) {
  const int fd = ::open(cachePath.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st {};
  if (fstat(fd, &st) != 0 || size_t(st.st_size) < sizeof(Header)) {
    ::close(fd);
    return false;
  }
  void *p = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return false;
  const size_t length = st.st_size;
  std::shared_ptr<const void> mapping(p, [length](const void *q) { munmap(const_cast<void *>(q), length); });
  const Header &h = *static_cast<const Header *>(p);
  if (std::memcmp(h.magic, kMagic, 8) != 0 || h.version != kFormatVersion || h.headerBytes != sizeof(Header) ||
      h.totalBytes != length || h.sourceSize != sourceSize || h.sourceMtime != sourceMtime)
    return false;
  m.dim = h.dim;
  m.heads = h.heads;
  m.ffnDim = h.ffnDim;
  m.encDepth = h.encDepth;
  m.decDepth = h.decDepth;
  m.srcEmbed = {nullptr, h.srcEmbedScale, h.srcVocab, h.dim};
  m.trgEmbed = {nullptr, h.trgEmbedScale, h.trgVocab, h.dim};
  m.outQuantMultA = h.outQuantMultA;
  m.outScale = h.outScale;
  // Size the layout these dimensions imply before touching anything past the header, so a
  // corrupt or outdated file is rebuilt instead of read out of bounds.
  {
    Model probe = m;
    Arena sizing(nullptr);
    sizing.take<Header>(1);
    Layout(probe, sizing, nullptr).run(h.tied != 0, "", "");
    if (((sizing.size() + 4095) & ~size_t(4095)) != length) return false;
  }
  Arena arena(static_cast<char *>(p));
  arena.take<Header>(1);
  Layout(m, arena, nullptr).run(h.tied != 0, "", "");
  m.storage = std::move(mapping);
  m.storageBytes = length;
  m.mapped = true;
  m.positions = sinusoidalPositions(m.dim);
  return true;
}

bool writeFile(const std::string &path, const void *data, size_t size) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return false;
  const char *p = static_cast<const char *>(data);
  size_t left = size;
  while (left > 0) {
    const ssize_t n = ::write(fd, p, left);
    if (n <= 0) {
      ::close(fd);
      ::unlink(tmp.c_str());
      return false;
    }
    p += n;
    left -= size_t(n);
  }
  if (::fsync(fd) != 0 || ::close(fd) != 0 || ::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    return false;
  }
  return true;
}

}  // namespace

void Model::pageOut() const {
#ifdef MADV_PAGEOUT
  if (mapped && storage) madvise(const_cast<void *>(storage.get()), storageBytes, MADV_PAGEOUT);
#endif
}

void Embedding::lookup(int32_t id, float *out) const {
  const int8_t *row = rows + size_t(id) * dim;
  for (int i = 0; i < dim; ++i) out[i] = float(row[i]) * scale;
}

Model Model::load(const std::string &path, const std::string &cachePath) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot open " + path);
  const uint64_t sourceSize = uint64_t(st.st_size);
  const int64_t sourceMtime = int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;

  Model m;
  if (!cachePath.empty() && mapCache(m, cachePath, sourceSize, sourceMtime)) return m;

  const File f = parse(path);
  require(f, "type", "transformer");
  require(f, "dec-cell", "ssru");
  require(f, "transformer-decoder-autoreg", "rnn");
  require(f, "transformer-postprocess", "dan");
  require(f, "transformer-postprocess-emb", "d");
  require(f, "transformer-ffn-activation", "relu");
  require(f, "transformer-ffn-depth", "2");
  require(f, "transformer-train-position-embeddings", "false");
  m.dim = f.intConfig("dim-emb");
  m.heads = f.intConfig("transformer-heads");
  m.ffnDim = f.intConfig("transformer-dim-ffn");
  m.encDepth = f.intConfig("enc-depth");
  m.decDepth = f.intConfig("dec-depth");
  if (m.dim % 64 || m.ffnDim % 64 || m.dim % m.heads) throw std::runtime_error("unsupported model dimensions");
  const bool tied = f.strConfig("tied-embeddings-all") == "true";
  const std::string srcName = tied ? "Wemb" : "encoder_Wemb", trgName = tied ? "Wemb" : "decoder_Wemb";
  {
    const Int8Matrix e = int8Matrix(f, trgName);
    // The output layer's activation scale is "none_QuantMultA", or (same value) the 1x1 int8
    // "<emb>_QuantMultA", stored as the byte 127 followed by its own scale s: 127 / s.
    if (f.has("none_QuantMultA")) {
      m.outQuantMultA = f.scalar("none_QuantMultA");
    } else {
      const Int8Matrix a = int8Matrix(f, trgName + "_QuantMultA");
      m.outQuantMultA = float(a.values[0]) / a.quantMult;
    }
    m.outScale = 1.0f / (m.outQuantMultA * e.quantMult);
  }

  // Size the block, then build it.
  Arena sizing(nullptr);
  sizing.take<Header>(1);
  {
    Model probe = m;
    Layout(probe, sizing, &f).run(tied, srcName, trgName);
  }
  const size_t total = (sizing.size() + 4095) & ~size_t(4095);
  auto block = std::shared_ptr<char>(static_cast<char *>(std::aligned_alloc(4096, total)), std::free);
  if (!block) throw std::bad_alloc();
  std::memset(block.get(), 0, total);
  Arena arena(block.get());
  Header &h = *arena.take<Header>(1);
  Layout(m, arena, &f).run(tied, srcName, trgName);
  std::memcpy(h.magic, kMagic, 8);
  h.version = kFormatVersion;
  h.headerBytes = sizeof(Header);
  h.totalBytes = total;
  h.sourceSize = sourceSize;
  h.sourceMtime = sourceMtime;
  h.dim = m.dim;
  h.heads = m.heads;
  h.ffnDim = m.ffnDim;
  h.encDepth = m.encDepth;
  h.decDepth = m.decDepth;
  h.srcVocab = m.srcEmbed.vocab;
  h.trgVocab = m.trgEmbed.vocab;
  h.tied = tied;
  h.srcEmbedScale = m.srcEmbed.scale;
  h.trgEmbedScale = m.trgEmbed.scale;
  h.outQuantMultA = m.outQuantMultA;
  h.outScale = m.outScale;

  // Prefer the mapped file over the heap copy: same bytes, but clean, shared, evictable pages.
  if (!cachePath.empty() && writeFile(cachePath, block.get(), total)) {
    Model mapped;
    if (mapCache(mapped, cachePath, sourceSize, sourceMtime)) return mapped;
  }
  m.storage = std::move(block);
  m.storageBytes = total;
  m.positions = sinusoidalPositions(m.dim);
  return m;
}

}  // namespace reseam::mt
