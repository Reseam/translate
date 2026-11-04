#include "text.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <map>
#include <mutex>
#include <fstream>
#include <stdexcept>

#include "sentencepiece_processor.h"

namespace reseam::mt {

// ---- Vocab --------------------------------------------------------------------------------

Vocab::Vocab(const std::string &path) : spm_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
  auto status = spm_->Load(path);
  if (!status.ok()) throw std::runtime_error("cannot load vocabulary " + path + ": " + status.ToString());
  size_ = spm_->GetPieceSize();
  eos_ = spm_->eos_id() >= 0 ? spm_->eos_id() : spm_->PieceToId("</s>");
  unk_ = spm_->unk_id();
}

Vocab::~Vocab() = default;

std::shared_ptr<Vocab> Vocab::shared(const std::string &path) {
  // Key: size and FNV-1a hash of the contents (a vocabulary is about 1 MB).
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::string bytes(size_t(in.tellg()), '\0');
  in.seekg(0);
  in.read(bytes.data(), std::streamsize(bytes.size()));
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : bytes) h = (h ^ c) * 1099511628211ull;
  const std::string key = std::to_string(bytes.size()) + ":" + std::to_string(h);
  static std::mutex mu;
  static std::map<std::string, std::weak_ptr<Vocab>> loaded;
  std::lock_guard<std::mutex> lock(mu);
  if (auto v = loaded[key].lock()) return v;
  auto v = std::make_shared<Vocab>(path);
  loaded[key] = v;
  return v;
}

std::vector<int32_t> Vocab::encode(std::string_view text) const {
  std::vector<int> ids;
  spm_->Encode(text, &ids);
  return {ids.begin(), ids.end()};
}

std::string Vocab::decode(const std::vector<int32_t> &ids) const {
  std::string out;
  spm_->Decode(std::vector<int>(ids.begin(), ids.end()), &out);
  return out;
}

// ---- Shortlist ----------------------------------------------------------------------------

Shortlist::Shortlist(const std::string &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot read " + path);
  }
  const size_t size = size_t(st.st_size);
  void *p = size ? mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
  ::close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  mapping_ = std::shared_ptr<const void>(p, [size](const void *q) { munmap(const_cast<void *>(q), size); });
  size_ = size;
  const char *data = static_cast<const char *>(p);
  struct Header {
    uint64_t magic, checksum, firstNum, bestNum, wordToOffsetSize, shortListsSize;
  } h;
  if (size < sizeof(h)) throw std::runtime_error(path + ": too short for a shortlist");
  std::memcpy(&h, data, sizeof(h));
  if (h.magic != 0xF11A48D5013417F5ull) throw std::runtime_error(path + ": not a binary shortlist");
  if (sizeof(h) + h.wordToOffsetSize * 8 + h.shortListsSize * 4 != size)
    throw std::runtime_error(path + ": shortlist size does not match its header");
  firstNum_ = h.firstNum;
  numOffsets_ = h.wordToOffsetSize;
  offsets_ = reinterpret_cast<const uint64_t *>(data + sizeof(h));
  lists_ = reinterpret_cast<const uint32_t *>(data + sizeof(h) + h.wordToOffsetSize * 8);
  // Every entry is checked once here, so a corrupt file cannot send a lookup out of bounds.
  for (uint64_t i = 0; i + 1 < numOffsets_; ++i)
    if (offsets_[i] > offsets_[i + 1] || offsets_[i + 1] > h.shortListsSize)
      throw std::runtime_error(path + ": corrupt shortlist offsets");
}

void Shortlist::pageOut() const {
#ifdef MADV_PAGEOUT
  madvise(const_cast<void *>(mapping_.get()), size_, MADV_PAGEOUT);
#endif
}

std::vector<int32_t> Shortlist::candidates(const int32_t *src, int n, bool sharedVocab, int trgVocab) const {
  std::vector<uint64_t> mask((trgVocab + 63) / 64, 0);
  auto set = [&](uint32_t id) {
    if (id < uint32_t(trgVocab)) mask[id >> 6] |= uint64_t(1) << (id & 63);
  };
  for (uint64_t i = 0; i < firstNum_ && int(i) < trgVocab; ++i) set(uint32_t(i));
  for (int i = 0; i < n; ++i) {
    const uint32_t w = uint32_t(src[i]);
    if (sharedVocab) set(w);
    if (w + 1 >= numOffsets_) continue;
    for (uint64_t j = offsets_[w]; j < offsets_[w + 1]; ++j) set(lists_[j]);
  }
  std::vector<int32_t> ids;
  for (size_t w = 0; w < mask.size(); ++w)
    for (uint64_t bits = mask[w]; bits; bits &= bits - 1) ids.push_back(int32_t(w * 64 + __builtin_ctzll(bits)));
  return ids;
}

// ---- Sentence splitting -------------------------------------------------------------------

namespace {

// Decodes the UTF-8 code point at s[i] (input is validated before it gets here).
char32_t codePoint(std::string_view s, size_t i, size_t *len) {
  const unsigned char c = s[i];
  if (c < 0x80) return *len = 1, c;
  if (c < 0xE0) return *len = 2, char32_t(c & 0x1F) << 6 | (s[i + 1] & 0x3F);
  if (c < 0xF0) return *len = 3, char32_t(c & 0x0F) << 12 | char32_t(s[i + 1] & 0x3F) << 6 | (s[i + 2] & 0x3F);
  return *len = 4, char32_t(c & 0x07) << 18 | char32_t(s[i + 1] & 0x3F) << 12 | char32_t(s[i + 2] & 0x3F) << 6 |
                       (s[i + 3] & 0x3F);
}

bool isSpace(char32_t c) { return c == ' ' || c == '\t' || c == '\r' || c == 0xA0 || c == 0x3000 || (c >= 0x2000 && c <= 0x200A); }
bool isLatinTerminator(char32_t c) { return c == '.' || c == '!' || c == '?' || c == 0x2026 /* … */; }
// Full-width terminators end a sentence without needing a space after them.
bool isWideTerminator(char32_t c) { return c == 0x3002 || c == 0xFF01 || c == 0xFF1F || c == 0xFF0E || c == 0x061F /* ؟ */ || c == 0x0964 /* । */; }
bool isCloser(char32_t c) {
  return c == '"' || c == '\'' || c == ')' || c == ']' || c == 0x201D || c == 0x2019 || c == 0xBB || c == 0x300D ||
         c == 0x300F || c == 0xFF09;
}
// Lowercase ASCII after ". " is usually an abbreviation ("e.g. this"), so no split there.
bool startsSentence(char32_t c) { return !(c >= 'a' && c <= 'z'); }

}  // namespace

namespace {

// Splits one run of text without URLs into sentences (appending to `out`).
void splitPlain(std::string_view text, std::vector<Segment> &out);

bool isUrlStart(std::string_view text, size_t i) {
  if (i > 0 && !std::isspace(static_cast<unsigned char>(text[i - 1])) && text[i - 1] != '(' && text[i - 1] != '"')
    return false;
  const std::string_view rest = text.substr(i);
  return rest.rfind("http://", 0) == 0 || rest.rfind("https://", 0) == 0 || rest.rfind("www.", 0) == 0;
}

}  // namespace

std::vector<Segment> splitSentences(std::string_view text) {
  std::vector<Segment> out;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (!isUrlStart(text, i)) continue;
    size_t end = i;
    while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) ++end;
    // Trailing punctuation belongs to the sentence, not the URL: "see https://x.y/z."
    while (end > i && std::strchr(".,;:!?)\"'", text[end - 1])) --end;
    splitPlain(text.substr(start, i - start), out);
    out.push_back({text.substr(i, end - i), false});
    start = i = end;
    if (i == text.size()) break;
  }
  splitPlain(text.substr(start), out);
  return out;
}

std::string restoreMentions(std::string_view source, std::string translation) {
  auto mentions = [](std::string_view s) {
    std::vector<std::pair<size_t, size_t>> found;  // offset, length
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] != '@' || (i > 0 && (std::isalnum(static_cast<unsigned char>(s[i - 1])) || s[i - 1] == '_'))) continue;
      size_t e = i + 1;
      while (e < s.size() && (std::isalnum(static_cast<unsigned char>(s[e])) || s[e] == '_')) ++e;
      if (e > i + 1) found.push_back({i, e - i});
      i = e;
    }
    return found;
  };
  const auto src = mentions(source), out = mentions(translation);
  if (src.empty() || src.size() != out.size()) return translation;
  std::string result;
  size_t at = 0;
  for (size_t k = 0; k < out.size(); ++k) {
    result.append(translation, at, out[k].first - at);
    result.append(source.substr(src[k].first, src[k].second));
    at = out[k].first + out[k].second;
  }
  result.append(translation, at);
  return result;
}

namespace {

void splitPlain(std::string_view text, std::vector<Segment> &out) {
  auto emit = [&](size_t b, size_t e, bool translate) {
    if (e > b) out.push_back({text.substr(b, e - b), translate});
  };
  size_t lineStart = 0;
  while (lineStart <= text.size()) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == std::string_view::npos) lineEnd = text.size();
    // Leading/trailing whitespace of each sentence is kept, not translated.
    size_t pos = lineStart, sentStart = lineStart;
    auto flush = [&](size_t end) {
      size_t b = sentStart, e = end;
      while (b < e && (text[b] == ' ' || text[b] == '\t' || text[b] == '\r')) ++b;
      while (e > b && (text[e - 1] == ' ' || text[e - 1] == '\t' || text[e - 1] == '\r')) --e;
      emit(sentStart, b, false);
      emit(b, e, true);
      emit(e, end, false);
      sentStart = end;
    };
    while (pos < lineEnd) {
      size_t len;
      const char32_t c = codePoint(text, pos, &len);
      pos += len;
      const bool latin = isLatinTerminator(c), wide = isWideTerminator(c);
      if (!latin && !wide) continue;
      // Absorb repeated terminators and closing quotes/brackets: `?!"`, `。」`.
      while (pos < lineEnd) {
        const char32_t d = codePoint(text, pos, &len);
        if (!(isLatinTerminator(d) || isWideTerminator(d) || isCloser(d))) break;
        pos += len;
      }
      size_t next = pos;
      bool space = false;
      while (next < lineEnd) {
        const char32_t d = codePoint(text, next, &len);
        if (!isSpace(d)) break;
        space = true;
        next += len;
      }
      if (next >= lineEnd) break;  // end of line ends the sentence anyway
      if (wide || (space && startsSentence(codePoint(text, next, &len)))) flush(pos);
    }
    flush(lineEnd);
    if (lineEnd < text.size()) emit(lineEnd, lineEnd + 1, false);  // the newline itself
    lineStart = lineEnd + 1;
  }
}

}  // namespace
}  // namespace reseam::mt
