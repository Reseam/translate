// Tokenization, shortlists and sentence splitting around the model.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace reseam::mt {

/// SentencePiece vocabulary as Marian uses it: ids are SentencePiece ids, EOS appended to sources.
class Vocab {
 public:
  explicit Vocab(const std::string &path);
  /// Loads `path`, or returns the already loaded vocabulary with identical contents: the two
  /// directions of a language pair usually share one vocabulary file.
  static std::shared_ptr<Vocab> shared(const std::string &path);
  ~Vocab();
  std::vector<int32_t> encode(std::string_view text) const;  // without EOS
  std::string decode(const std::vector<int32_t> &ids) const;
  int32_t eos() const { return eos_; }
  int32_t unk() const { return unk_; }
  int size() const { return size_; }

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_;
  int32_t eos_ = 0, unk_ = 1;
  int size_ = 0;
};

/// Marian's binary lexical shortlist: the target words a sentence may produce are the `firstNum`
/// most frequent ones plus, for every source word, its `bestNum` likeliest translations (and the
/// word itself when source and target share a vocabulary).
class Shortlist {
 public:
  explicit Shortlist(const std::string &path);
  /// The target words `src` may translate to, sorted, without duplicates.
  std::vector<int32_t> candidates(const int32_t *src, int n, bool sharedVocab, int trgVocab) const;
  /// Drops the mapped file's pages now (see Model::pageOut).
  void pageOut() const;

 private:
  std::shared_ptr<const void> mapping_;  // the file, memory-mapped (clean, evictable pages)
  size_t size_ = 0;
  uint64_t firstNum_ = 0;
  const uint64_t *offsets_ = nullptr;
  uint64_t numOffsets_ = 0;
  const uint32_t *lists_ = nullptr;
};

/// A piece of the input: a sentence to translate, or text kept as it is (spacing, line breaks).
struct Segment {
  std::string_view text;
  bool translate;
};

/// Splits text into sentences, keeping everything between them verbatim so the translation can
/// be put back together with the original line breaks and spacing. URLs are kept verbatim too:
/// the models would "translate" them, or run on repeating them.
std::vector<Segment> splitSentences(std::string_view text);

/// Puts the source's @mentions back into a translation: models turn "@usuario" into "@user".
/// Only when source and translation hold the same number of mentions, matched in order;
/// otherwise the translation is returned unchanged.
std::string restoreMentions(std::string_view source, std::string translation);

}  // namespace reseam::mt
