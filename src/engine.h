// Translation engine: loads Firefox Translations models and serves translations from a pool of
// worker threads with continuous batching (sentences join and leave a running decode batch at
// every step, so a new request never waits for a whole batch to finish).
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reseam::mt {

struct EngineOptions {
  std::filesystem::path modelsDir;  // one folder per direction: <from>-<to>/{model.*.bin, lex.*.bin, *vocab*.spm}
  int workers = 1;                  // threads in the compute team; one per physical core
  int maxBatch = 64;                // sentences decoded together per model
  bool preload = false;             // load every model at startup instead of on first use
  bool cacheWeights = true;         // keep packed weights in <model dir>/model.reseam and map them
  int idleSeconds = 300;            // drop a model's weights from memory after this long unused
                                    // (they reload from disk on the next request); 0: never
  int firstCore = -1;               // >= 0: pin team thread i to core firstCore + i (keeps each core's
                                    // weight slices in its L2, and keeps lower cores for I/O threads)
  int rowModeRows = 0;              // batch size from which decoder steps split rows instead of columns
                                    // (0: 4 per thread)
};

/// Bad input from the caller (unsupported pair, invalid UTF-8), as opposed to a failure on our side.
struct InputError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct EngineStats {
  uint64_t sentences = 0;       // sentences translated (each pivot leg counts)
  uint64_t tokens = 0;          // target tokens generated
  uint64_t decodeSteps = 0;     // batched decoder steps run
  uint64_t decodeRows = 0;      // sentence-steps decoded (decodeRows / decodeSteps = mean batch)
  uint64_t encodedTokens = 0;   // source tokens encoded
  uint64_t rowJobs = 0;         // team jobs run in row mode (throughput) ...
  uint64_t columnJobs = 0;      // ... and in column mode (latency)
};

class Engine {
 public:
  explicit Engine(const EngineOptions &options);
  ~Engine();

  /// Translates each text, keeping order, line breaks and spacing between sentences.
  /// Thread-safe; blocks until done. Throws InputError for bad input.
  std::vector<std::string> translate(const std::string &from, const std::string &to,
                                     const std::vector<std::string> &texts);

  /// Directions available on disk, as (from, to).
  std::vector<std::pair<std::string, std::string>> pairs() const;
  /// Whether from -> to works, directly or through English.
  bool supports(const std::string &from, const std::string &to) const;

  EngineStats stats() const;
  const char *isa() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Strict UTF-8 check (no overlongs, surrogates, or code points past U+10FFFF).
bool isValidUtf8(std::string_view s);

}  // namespace reseam::mt
