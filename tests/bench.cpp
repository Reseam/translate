// Benchmarks the engine on a TSV corpus (from \t to \t text, with \n and \\ escaped), the same two
// measurements used for the Bergamot baselines:
//   1. latency: the first 200 posts one at a time (what a single user waits);
//   2. throughput: every post submitted at once from `clients` threads.
// Writes translations to <out> in the same TSV form for quality scoring.
//
// usage: bench <models> <corpus.tsv> <n> <workers> <out.tsv> [clients=64] [maxBatch=32]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <thread>

#include <pthread.h>
#include <sched.h>

#include "engine.h"

using Clock = std::chrono::steady_clock;

namespace {

struct Row {
  std::string from, to, text;
};

std::string unescape(const std::string &s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      ++i;
      o += s[i] == 'n' ? '\n' : s[i];
    } else {
      o += s[i];
    }
  }
  return o;
}

std::string escape(const std::string &s) {
  std::string o;
  for (char c : s) o += c == '\n' ? std::string("\\n") : c == '\\' ? std::string("\\\\") : std::string(1, c);
  return o;
}

// Resident memory in MB: "VmRSS" (all), "RssAnon" (private heap), "RssFile" (mapped files).
long rssMb(const char *field = "VmRSS:") {
  std::ifstream f("/proc/self/status");
  const std::string key = field;
  for (std::string l; std::getline(f, l);)
    if (l.rfind(key, 0) == 0) return std::stol(l.substr(key.size())) / 1024;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: bench <models> <corpus.tsv> <n> <workers> <out.tsv> [clients] [maxBatch]\n");
    return 2;
  }
  reseam::mt::EngineOptions opt;
  opt.modelsDir = argv[1];
  const size_t n = std::stoul(argv[3]);
  opt.workers = std::stoi(argv[4]);
  const size_t clients = argc > 6 ? std::stoul(argv[6]) : 64;
  if (argc > 7) opt.maxBatch = std::stoi(argv[7]);
  // RESEAM_MT_PIN=<first core>: pin the team from that core up, and keep this process's other
  // threads (the clients) on the cores below it.
  if (const char *pin = std::getenv("RESEAM_MT_PIN")) {
    opt.firstCore = std::stoi(pin);
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < std::max(1, opt.firstCore); ++c) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
  if (const char *rm = std::getenv("RESEAM_MT_ROWMODE")) opt.rowModeRows = std::stoi(rm);
  opt.preload = std::getenv("BENCH_PRELOAD") != nullptr;  // every model in the folder, as a server would

  std::vector<Row> rows;
  std::ifstream in(argv[2]);
  for (std::string line; rows.size() < n && std::getline(in, line);) {
    auto a = line.find('\t'), b = line.find('\t', a + 1);
    rows.push_back({line.substr(0, a), line.substr(a + 1, b - a - 1), unescape(line.substr(b + 1))});
  }
  const long rssBefore = rssMb();
  reseam::mt::Engine engine(opt);

  auto t0 = Clock::now();
  std::set<std::pair<std::string, std::string>> pairs;
  for (auto &r : rows) pairs.insert({r.from, r.to});
  for (auto &[f, t] : pairs) engine.translate(f, t, {"Hi."});
  const double loadMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  std::vector<double> lat;
  const size_t latencyPosts = std::getenv("BENCH_SKIP_LATENCY") ? 1 : 200;
  for (size_t i = 0; i < std::min<size_t>(latencyPosts, rows.size()); ++i) {
    auto s = Clock::now();
    engine.translate(rows[i].from, rows[i].to, {rows[i].text});
    lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());
  }
  std::sort(lat.begin(), lat.end());
  auto q = [&](double p) { return lat[std::min(lat.size() - 1, size_t(p * lat.size()))]; };
  double mean = 0;
  for (double v : lat) mean += v / lat.size();

  const auto before = engine.stats();
  std::vector<std::string> out(rows.size());
  std::atomic<size_t> next{0};
  auto s = Clock::now();
  std::vector<std::thread> threads;
  for (size_t c = 0; c < clients; ++c)
    threads.emplace_back([&] {
      for (size_t i; (i = next++) < rows.size();) out[i] = engine.translate(rows[i].from, rows[i].to, {rows[i].text})[0];
    });
  for (auto &t : threads) t.join();
  const double secs = std::chrono::duration<double>(Clock::now() - s).count();
  size_t bytes = 0;
  for (auto &r : rows) bytes += r.text.size();

  std::ofstream o(argv[5]);
  for (size_t i = 0; i < rows.size(); ++i) o << rows[i].from << '\t' << rows[i].to << '\t' << escape(out[i]) << '\n';
  auto st = engine.stats();
  st.decodeSteps -= before.decodeSteps;
  st.tokens -= before.tokens;
  st.encodedTokens -= before.encodedTokens;
  st.decodeRows -= before.decodeRows;
  st.rowJobs -= before.rowJobs;
  st.columnJobs -= before.columnJobs;
  std::printf(
      "{\"engine\":\"reseam-mt\",\"isa\":\"%s\",\"workers\":%d,\"maxBatch\":%d,\"clients\":%zu,\"posts\":%zu,"
      "\"load_ms\":%.0f,\"latency_ms\":{\"mean\":%.2f,\"p50\":%.2f,\"p90\":%.2f,\"p99\":%.2f},"
      "\"throughput\":{\"posts_per_s\":%.1f,\"bytes_per_s\":%.0f,\"secs\":%.2f},"
      "\"mean_batch\":%.1f,\"out_tokens_per_s\":%.0f,\"in_tokens_per_s\":%.0f,\"row_jobs\":%llu,\"column_jobs\":%llu,\"rss_mb\":%ld,\"anon_mb\":%ld,\"file_mb\":%ld}\n",
      engine.isa(), opt.workers, opt.maxBatch, clients, rows.size(), loadMs, mean, q(0.5), q(0.9), q(0.99),
      rows.size() / secs, bytes / secs, secs, st.decodeSteps ? double(st.decodeRows) / st.decodeSteps : 0.0, st.tokens / secs, st.encodedTokens / secs,
      (unsigned long long)st.rowJobs, (unsigned long long)st.columnJobs, rssMb() - rssBefore, rssMb("RssAnon:"),
      rssMb("RssFile:"));
}
