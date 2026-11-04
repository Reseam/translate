#include "engine.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <thread>

#include "batch.h"
#include "gemm.h"
#include "team.h"
#include "text.h"

namespace reseam::mt {

namespace fs = std::filesystem;

namespace {

constexpr int kMaxSourceTokens = 127;
// Source tokens admitted per scheduler turn, across all models: bounds the encoder's working
// memory and keeps one huge request from stalling everyone else's decoding for long.
constexpr int kMaxAdmitTokens = 4096;  // longer sentences are cut into pieces (Bergamot: max-length-break - 1)
constexpr const char *kPivot = "en";

std::pair<std::string, std::string> splitPair(const std::string &name) {
  if (name.rfind("en-", 0) == 0) return {"en", name.substr(3)};
  if (name.size() > 3 && name.compare(name.size() - 3, 3, "-en") == 0) return {name.substr(0, name.size() - 3), "en"};
  return {"", ""};
}

// Language codes as clients send them (X, Telegram, Android, BCP 47) to the codes the Firefox
// models use: lowercase primary subtag, script subtags for Chinese, and legacy ISO 639 aliases.
std::string normalizeLanguage(std::string code) {
  for (auto &c : code) c = (c == '_') ? '-' : char(std::tolower(static_cast<unsigned char>(c)));
  if (code == "zh-hant" || code == "zh-tw" || code == "zh-hk" || code == "zh-mo" || code.rfind("zh-hant-", 0) == 0)
    return "zh-Hant";
  if (code == "zh" || code.rfind("zh-", 0) == 0) return "zh-Hans";
  const std::string primary = code.substr(0, code.find('-'));
  if (primary == "in") return "id";  // Indonesian (X still sends "in")
  if (primary == "iw") return "he";  // Hebrew
  if (primary == "no") return "nb";  // Norwegian Bokmål
  if (primary == "fil") return "tl"; // Filipino
  return primary;
}

bool hasModel(const fs::path &dir) {
  for (const auto &e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().filename().string().rfind("model.", 0) == 0 &&
        e.path().extension() == ".bin")
      return true;
  return false;
}

}  // namespace

// ---- Requests and sentences ---------------------------------------------------------------

struct Job;

struct Request {
  // Per sentence: target ids, decoded by the requester. A sentence normally has one part; a
  // pivoted one whose English is too long is translated in several (joined with spaces).
  std::vector<std::vector<std::vector<int32_t>>> outputs;
  std::vector<std::unique_ptr<Job>> extra;  // jobs created for those extra parts (scheduler only)
  std::atomic<int> remaining{0};
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;  // set if any sentence failed

  void finishOne() {
    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(mu);
      done = true;
      cv.notify_all();
    }
  }
  void fail(const std::string &message) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (error.empty()) error = message;
    }
    finishOne();
  }
};

struct Runtime;

struct Job : Sentence {
  Runtime *rt;    // model for the current leg
  Runtime *next;  // second leg when pivoting through English, else null
  Request *req;
  int index;      // slot in req->outputs
  int part = 0;   // and part within it
};

// A loaded model, the sentences waiting for it, and its running batch (scheduler-owned).
struct Runtime {
  std::string from, to;
  Bundle bundle;
  std::deque<Job *> pending;  // guarded by Engine::Impl::mu
  std::unique_ptr<Batch> batch;
  std::chrono::steady_clock::time_point lastUsed = std::chrono::steady_clock::now();  // scheduler only
  bool pagedOut = false;                                                               // scheduler only
  Runtime(const std::string &f, const std::string &t, const fs::path &dir, bool cache)
      : from(f), to(t), bundle(dir.string(), cache ? (dir / "model.reseam").string() : std::string()) {}
};

// ---- Engine -------------------------------------------------------------------------------

struct Engine::Impl {
  struct Pair {
    fs::path dir;
    std::mutex mu;
    std::unique_ptr<Runtime> rt;
    std::string error;  // last load failure; retried after kRetryAfter
    std::chrono::steady_clock::time_point failedAt;
  };
  static constexpr auto kRetryAfter = std::chrono::seconds(30);

  EngineOptions opt;
  std::map<std::string, std::unique_ptr<Pair>> pairs;  // "from|to"; fixed after construction

  std::mutex mu;  // guards everything below and Runtime::pending
  std::condition_variable cv;
  std::vector<Runtime *> runtimes;
  bool stopping = false;
  EngineCounters counters;
  std::thread scheduler;

  explicit Impl(const EngineOptions &o) : opt(o) {
    if (!fs::is_directory(opt.modelsDir)) throw std::runtime_error("models folder not found: " + opt.modelsDir.string());
    for (const auto &e : fs::directory_iterator(opt.modelsDir)) {
      if (!e.is_directory()) continue;
      auto [from, to] = splitPair(e.path().filename().string());
      if (from.empty() || !hasModel(e.path())) continue;
      // Folders use whatever codes their source used ("zh-en" in older mirrors, "zh-Hans-en" from
      // Mozilla); requests are normalized, so folder names are too.
      from = normalizeLanguage(from);
      to = normalizeLanguage(to);
      auto p = std::make_unique<Pair>();
      p->dir = e.path();
      pairs.emplace(from + "|" + to, std::move(p));
    }
    if (pairs.empty()) throw std::runtime_error("no models in " + opt.modelsDir.string());
    if (opt.workers < 1) opt.workers = 1;
    if (opt.maxBatch < 1) opt.maxBatch = 1;
    if (opt.rowModeRows <= 0) opt.rowModeRows = 4 * opt.workers;
    activeIsa();  // detect the CPU (and ask the kernel for AMX) before any thread starts
    scheduler = std::thread([this] { run(); });
    if (opt.preload)
      for (auto &[k, p] : pairs) {
        try {
          load(k.substr(0, k.find('|')), k.substr(k.find('|') + 1));
        } catch (const std::exception &e) {
          std::fprintf(stderr, "reseam-translate: %s\n", e.what());  // the others still load
        }
      }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mu);
      stopping = true;
    }
    cv.notify_all();
    scheduler.join();
  }

  bool has(const std::string &from, const std::string &to) const { return pairs.count(from + "|" + to) > 0; }

  Runtime *load(const std::string &from, const std::string &to) {
    Pair &p = *pairs.at(from + "|" + to);
    std::lock_guard<std::mutex> pairLock(p.mu);
    if (p.rt) return p.rt.get();
    // A failure (a file being replaced, out of memory) is retried, but not on every request.
    if (!p.error.empty() && std::chrono::steady_clock::now() - p.failedAt < kRetryAfter)
      throw std::runtime_error("could not load " + from + "-" + to + ": " + p.error);
    try {
      auto rt = std::make_unique<Runtime>(from, to, p.dir, opt.cacheWeights);
      {
        std::lock_guard<std::mutex> lock(mu);
        runtimes.push_back(rt.get());
      }
      p.rt = std::move(rt);
      p.error.clear();
    } catch (const std::exception &e) {
      p.error = e.what();
      p.failedAt = std::chrono::steady_clock::now();
      throw std::runtime_error("could not load " + from + "-" + to + ": " + p.error);
    }
    return p.rt.get();
  }

  void enqueue(const std::vector<Job *> &jobs) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!stopping) {
        for (auto *j : jobs) j->rt->pending.push_back(j);
        cv.notify_all();
        return;
      }
    }
    for (auto *j : jobs) j->req->fail("the translation engine is shutting down");
  }

  // Finishes a sentence: hand it to its second leg, or give the ids back to the requester.
  // Scheduler thread only.
  void complete(Active &a) {
    Job *j = static_cast<Job *>(a.sentence);
    if (j->next) {
      // Second leg: the English goes to the next model, in pieces if it is too long for one.
      Runtime *next = j->next;
      const Vocab &v = *next->bundle.srcVocab;
      const std::vector<int32_t> ids = v.encode(j->rt->bundle.trgVocab->decode(a.out));
      const size_t parts = std::max<size_t>(1, (ids.size() + kMaxSourceTokens - 1) / kMaxSourceTokens);
      Request &req = *j->req;
      req.outputs[j->index].resize(parts);
      req.remaining.fetch_add(int(parts) - 1, std::memory_order_acq_rel);
      std::vector<Job *> legs;
      for (size_t k = 0; k < parts; ++k) {
        Job *leg = j;
        if (k > 0) {
          req.extra.push_back(std::make_unique<Job>());
          leg = req.extra.back().get();
          leg->req = &req;
          leg->index = j->index;
          leg->part = int(k);
        }
        leg->src.assign(ids.begin() + std::min(k * kMaxSourceTokens, ids.size()),
                        ids.begin() + std::min((k + 1) * kMaxSourceTokens, ids.size()));
        leg->src.push_back(v.eos());
        leg->candidates = next->bundle.candidates(leg->src);
        leg->rt = next;
        leg->next = nullptr;
        legs.push_back(leg);
      }
      enqueue(legs);
      return;
    }
    j->req->outputs[j->index][j->part] = std::move(a.out);
    j->req->finishOne();
  }

  // Drops the weights of models unused for opt.idleSeconds from memory (scheduler thread, with
  // mu held). They stay loaded: the next request reads the pages back from the cache file.
  void pageOutIdle() {
    if (opt.idleSeconds <= 0) return;
    const auto now = std::chrono::steady_clock::now();
    for (Runtime *rt : runtimes)
      if (!rt->pagedOut && now - rt->lastUsed > std::chrono::seconds(opt.idleSeconds)) {
        rt->bundle.model.pageOut();
        rt->bundle.shortlist->pageOut();
        rt->pagedOut = true;
      }
  }

  // Fails every sentence the batches hold (after an error on the scheduler thread, or at
  // shutdown), so no requester waits forever.
  void failRunning(const std::string &message) {
    std::vector<Sentence *> lost;
    for (Runtime *rt : runtimes)
      if (rt->batch) {
        rt->batch->drain(lost);
        rt->batch.reset();
      }
    for (Sentence *s : lost) static_cast<Job *>(s)->req->fail(message);
  }

  // One piece of a team job: the encoder for newly admitted sentences of one model, or one
  // decoder step for its active sentences.
  struct Part {
    Batch *batch;
    bool encode;
    int rows;
  };
  struct Segment {
    size_t part;
    int r0, r1;
  };

  // Plans a row-mode job: lays the rows of all parts end to end, weights each by its cost, and
  // cuts the sequence into one contiguous range per thread of about equal cost. Encoder
  // sentences are never cut (self-attention needs a sentence's rows together); an encoder row
  // costs about 1.3 decoder rows (6 encoder layers vs 4 decoder layers + output layer).
  static std::vector<std::vector<Segment>> plan(const std::vector<Part> &parts, int threads) {
    struct Unit {
      size_t part;
      int r0, r1;
      double cost;
    };
    std::vector<Unit> units;
    double total = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      const Part &p = parts[i];
      if (p.encode) {
        const auto &rows = p.batch->sentenceRows();
        for (size_t s = 0; s + 1 < rows.size(); ++s) units.push_back({i, rows[s], rows[s + 1], 1.3 * (rows[s + 1] - rows[s])});
      } else {
        for (int r = 0; r < p.rows; ++r) units.push_back({i, r, r + 1, 1.0});
      }
    }
    for (auto &u : units) total += u.cost;
    std::vector<std::vector<Segment>> out(threads);
    double before = 0;
    for (const Unit &u : units) {
      // A unit goes to the thread whose share holds its midpoint.
      const int t = std::min(threads - 1, int((before + u.cost / 2) * threads / total));
      before += u.cost;
      auto &segs = out[t];
      if (!segs.empty() && segs.back().part == u.part && segs.back().r1 == u.r0)
        segs.back().r1 = u.r1;
      else
        segs.push_back({u.part, u.r0, u.r1});
    }
    return out;
  }

  // The scheduler: owns the team and every model's running batch. Each turn admits waiting
  // sentences and runs one team job holding both their encoding and one decoder step for the
  // active sentences of every model.
  //
  // Little work in total (the latency case): column mode, parts one after another, every core on
  // each product. Enough work: row mode, each thread gets a contiguous share of all rows of all
  // parts (see plan) and runs it start to finish without a single barrier.
  void run() {
    Team team(opt.workers, opt.firstCore);
    Scratch scratch;
    uint64_t turns = 0;
    std::vector<std::pair<Runtime *, std::vector<Sentence *>>> admitted;
    std::vector<Part> parts;
    std::vector<Active> done;
    EngineCounters local;
    for (;;) {
      admitted.clear();
      parts.clear();
      {
        std::unique_lock<std::mutex> lock(mu);
        counters.sentences += local.sentences;
        counters.tokens += local.tokens;
        counters.decodeSteps += local.decodeSteps;
        counters.decodeRows += local.decodeRows;
        counters.encodedTokens += local.encodedTokens;
        counters.rowJobs += local.rowJobs;
        counters.columnJobs += local.columnJobs;
        local = {};
        auto busy = [&] {
          for (Runtime *rt : runtimes)
            if (!rt->pending.empty() || (rt->batch && !rt->batch->empty())) return true;
          return false;
        };
        while (!stopping && !busy()) {
          scratch.release();  // idle: give the working memory back
          pageOutIdle();
          cv.wait_for(lock, std::chrono::seconds(opt.idleSeconds > 0 ? std::min(opt.idleSeconds, 60) : 3600));
        }
        if (stopping) {
          std::vector<Job *> queued;
          for (Runtime *rt : runtimes) {
            queued.insert(queued.end(), rt->pending.begin(), rt->pending.end());
            rt->pending.clear();
          }
          lock.unlock();
          for (Job *j : queued) j->req->fail("the translation engine is shutting down");
          failRunning("the translation engine is shutting down");
          return;
        }
        int budget = kMaxAdmitTokens;
        const auto now = std::chrono::steady_clock::now();
        for (Runtime *rt : runtimes)
          if (!rt->pending.empty() || (rt->batch && !rt->batch->empty())) {
            rt->lastUsed = now;
            rt->pagedOut = false;
          }
        if (++turns % 4096 == 0) pageOutIdle();  // busy servers check now and then too
        for (Runtime *rt : runtimes) {
          // Decode first: prepareStep must see the batch before newcomers join it.
          if (rt->batch && !rt->batch->empty()) parts.push_back({rt->batch.get(), false, 0});
          const int running = rt->batch ? rt->batch->size() : 0;
          std::vector<Sentence *> take;
          while (running + int(take.size()) < opt.maxBatch && !rt->pending.empty() &&
                 (budget >= int(rt->pending.front()->src.size()) || (budget == kMaxAdmitTokens && take.empty()))) {
            budget -= int(rt->pending.front()->src.size());
            take.push_back(rt->pending.front());
            rt->pending.pop_front();
          }
          if (!take.empty()) admitted.emplace_back(rt, std::move(take));
        }
      }
      // Everything here but the team job itself can throw (out of memory, a vocabulary error on
      // the pivot path). The job cannot: every buffer it touches is sized beforehand. On an error
      // the sentences in flight fail back to their requesters and the server carries on.
      try {
        double cost = 0;
        for (Part &p : parts) {
          p.rows = p.batch->prepareStep();
          cost += p.rows;
        }
        for (auto &[rt, take] : admitted) {
          if (!rt->batch) rt->batch = std::make_unique<Batch>(rt->bundle, team.size());
          const int rows = rt->batch->prepareAdmit(std::move(take));
          parts.push_back({rt->batch.get(), true, rows});
          cost += 1.3 * rows;
        }
        // Working memory for the whole job, from one shared block: size it, then hand it out.
        scratch.begin();
        for (const Part &p : parts) p.encode ? p.batch->admitBuffers(scratch) : p.batch->stepBuffers(scratch);
        scratch.commit();
        for (const Part &p : parts) p.encode ? p.batch->admitBuffers(scratch) : p.batch->stepBuffers(scratch);
        const bool rowMode = cost >= opt.rowModeRows && team.size() > 1;
        const auto segments = rowMode ? plan(parts, team.size()) : std::vector<std::vector<Segment>>();
        ++(rowMode ? local.rowJobs : local.columnJobs);
        team.run([&](int t, int n) {
          auto go = [&](const Part &p, const Split &sp) {
            if (p.encode)
              p.batch->encodeRows(sp);
            else
              p.batch->stepRows(sp);
          };
          if (rowMode) {
            for (const Segment &sg : segments[t]) go(parts[sg.part], Split{team, t, n, false, parts[sg.part].rows, sg.r0, sg.r1});
          } else {
            for (const Part &p : parts) {
              Split sp{team, t, n, true, p.rows, 0, 0};
              Team::split(p.rows, t, n, 1, sp.r0, sp.r1);
              go(p, sp);
            }
          }
        });
        done.clear();
        for (const Part &p : parts)
          if (!p.encode) p.batch->finishStep(done, local);
        for (const Part &p : parts)
          if (p.encode) p.batch->finishAdmit(local);
      } catch (const std::exception &e) {
        std::fprintf(stderr, "reseam-translate: %s\n", e.what());
        for (auto &[rt, take] : admitted)
          for (Sentence *s : take) static_cast<Job *>(s)->req->fail("translation failed");
        failRunning("translation failed");
        continue;
      }
      for (auto &a : done) {
        ++local.sentences;
        try {
          complete(a);
        } catch (const std::exception &e) {
          std::fprintf(stderr, "reseam-translate: %s\n", e.what());
          static_cast<Job *>(a.sentence)->req->fail("translation failed");
        }
      }
    }
  }
};

Engine::Engine(const EngineOptions &options) : impl_(std::make_unique<Impl>(options)) {}
Engine::~Engine() = default;

std::vector<std::pair<std::string, std::string>> Engine::pairs() const {
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto &[k, _] : impl_->pairs) out.emplace_back(k.substr(0, k.find('|')), k.substr(k.find('|') + 1));
  return out;
}

bool Engine::supports(const std::string &fromCode, const std::string &toCode) const {
  const std::string from = normalizeLanguage(fromCode), to = normalizeLanguage(toCode);
  if (from == to || impl_->has(from, to)) return true;
  return from != kPivot && to != kPivot && impl_->has(from, kPivot) && impl_->has(kPivot, to);
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  const auto &c = impl_->counters;
  return {c.sentences, c.tokens, c.decodeSteps, c.decodeRows, c.encodedTokens, c.rowJobs, c.columnJobs};
}

const char *Engine::isa() const { return isaName(activeIsa()); }

std::vector<std::string> Engine::translate(const std::string &fromCode, const std::string &toCode,
                                           const std::vector<std::string> &texts) {
  const std::string from = normalizeLanguage(fromCode), to = normalizeLanguage(toCode);
  for (const auto &t : texts)
    if (!isValidUtf8(t)) throw InputError("text is not valid UTF-8");
  if (from == to) return texts;
  if (!supports(from, to)) throw InputError("no model for " + from + " to " + to);
  Runtime *first, *second = nullptr;
  if (impl_->has(from, to)) {
    first = impl_->load(from, to);
  } else {
    first = impl_->load(from, kPivot);
    second = impl_->load(kPivot, to);
  }
  const Vocab &srcVocab = *first->bundle.srcVocab;
  const Vocab &trgVocab = *(second ? second : first)->bundle.trgVocab;

  // Each text becomes a list of pieces: kept verbatim, or the translation of job k.
  struct Piece {
    std::string_view verbatim;
    int job = -1;
  };
  std::vector<std::vector<Piece>> pieces(texts.size());
  std::vector<std::unique_ptr<Job>> jobs;
  Request req;
  for (size_t i = 0; i < texts.size(); ++i) {
    for (const Segment &seg : splitSentences(texts[i])) {
      if (!seg.translate) {
        pieces[i].push_back({seg.text, -1});
        continue;
      }
      const std::vector<int32_t> ids = srcVocab.encode(seg.text);
      // Overlong sentences are translated in pieces of kMaxSourceTokens tokens.
      for (size_t b = 0; b < std::max<size_t>(ids.size(), 1); b += kMaxSourceTokens) {
        if (b > 0) pieces[i].push_back({" ", -1});
        auto j = std::make_unique<Job>();
        j->rt = first;
        j->next = second;
        j->src.assign(ids.begin() + std::min(b, ids.size()), ids.begin() + std::min(b + kMaxSourceTokens, ids.size()));
        j->src.push_back(srcVocab.eos());
        j->candidates = first->bundle.candidates(j->src);  // here, not on the scheduler thread
        j->req = &req;
        j->index = int(jobs.size());
        pieces[i].push_back({{}, j->index});
        jobs.push_back(std::move(j));
      }
    }
  }
  if (!jobs.empty()) {
    req.outputs.assign(jobs.size(), std::vector<std::vector<int32_t>>(1));
    req.remaining = int(jobs.size());
    std::vector<Job *> raw;
    for (auto &j : jobs) raw.push_back(j.get());
    impl_->enqueue(raw);
    std::unique_lock<std::mutex> lock(req.mu);
    req.cv.wait(lock, [&] { return req.done; });
    if (!req.error.empty()) throw std::runtime_error(req.error);
  }
  std::vector<std::string> out(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    for (const Piece &p : pieces[i]) {
      if (p.job < 0) {
        out[i] += p.verbatim;
        continue;
      }
      const auto &parts = req.outputs[p.job];
      for (size_t k = 0; k < parts.size(); ++k) out[i] += (k ? " " : "") + trgVocab.decode(parts[k]);
    }
    out[i] = restoreMentions(texts[i], std::move(out[i]));
  }
  return out;
}

bool isValidUtf8(std::string_view s) {
  const auto *p = reinterpret_cast<const unsigned char *>(s.data());
  const auto *end = p + s.size();
  while (p < end) {
    const unsigned char c = *p;
    if (c < 0x80) {
      ++p;
      continue;
    }
    size_t len;
    uint32_t cp;
    if (c >= 0xC2 && c <= 0xDF) len = 2, cp = c & 0x1F;
    else if (c >= 0xE0 && c <= 0xEF) len = 3, cp = c & 0x0F;
    else if (c >= 0xF0 && c <= 0xF4) len = 4, cp = c & 0x07;
    else return false;
    if (size_t(end - p) < len) return false;
    for (size_t k = 1; k < len; ++k) {
      if ((p[k] & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (p[k] & 0x3F);
    }
    if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    p += len;
  }
  return true;
}

}  // namespace reseam::mt
