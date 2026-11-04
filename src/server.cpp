// reseam-translate: HTTP front of the engine.
//
//   GET  /health          {"status":"ok", "isa":..., "threads":..., "pairs":..., counters}
//   GET  /v1/languages    {"pairs":[["es","en"], ...], "pivot":"en"}
//   POST /v1/translate    {"from":"es","to":"en","texts":["..."]} -> {"translations":["..."]}
//
// Settings come from flags or the environment (flag names in parentheses):
//   MODELS_DIR (--models)      folder of <from>-<to>/ model folders              [./models]
//   HOST (--host), PORT (--port)                                               [0.0.0.0, 8080]
//   THREADS (--threads)        compute threads, one per physical core            [all cores]
//   MAX_BATCH (--max-batch)    sentences decoded together per model              [64]
//   MAX_TEXTS (--max-texts)    texts per request                                 [50]
//   MAX_CHARS (--max-chars)    bytes of text per request                         [20000]
//   HTTP_THREADS (--http-threads) concurrent requests held open                  [64]
//   PIN_FROM_CORE (--pin-from-core) pin compute thread i to core N + i           [off]
//   PRELOAD (--preload)        load every model at startup                       [off]
//   IDLE_SECONDS (--idle-seconds) drop an unused model's weights from memory      [300]
//                              after this long; 0 keeps everything resident
//   TRANSLATE_TOKEN            if set, requests must send "Authorization: Bearer <token>"
//
// Request texts are never logged.
#include <httplib.h>

#include <atomic>
#include <pthread.h>
#include <sched.h>

#include <csignal>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "engine.h"

using nlohmann::json;

namespace {

struct Settings {
  std::string models = "./models", host = "0.0.0.0", token;
  int port = 8080, threads = 0, maxBatch = 64, httpThreads = 64, pinFromCore = -1, idleSeconds = 300;
  size_t maxTexts = 50, maxChars = 20000;
  bool preload = false;
};

std::string env(const char *name, const std::string &fallback) {
  const char *v = std::getenv(name);
  return v && *v ? v : fallback;
}

// CPUs this process may actually use: its affinity mask, capped by a cgroup v2 CPU quota (a
// container with --cpus=2 on a 16-core host must not start 16 spinning threads).
int availableCpus() {
  int cpus = int(std::max(1u, std::thread::hardware_concurrency()));
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) cpus = std::max(1, CPU_COUNT(&set));
  if (FILE *f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
    char quota[32] = {};
    long period = 0;
    if (std::fscanf(f, "%31s %ld", quota, &period) == 2 && std::strcmp(quota, "max") != 0 && period > 0)
      cpus = std::max(1, std::min(cpus, int((std::atol(quota) + period - 1) / period)));
    std::fclose(f);
  }
  return cpus;
}

Settings parse(int argc, char **argv) {
  Settings s;
  s.models = env("MODELS_DIR", s.models);
  s.host = env("HOST", s.host);
  s.port = std::stoi(env("PORT", std::to_string(s.port)));
  s.threads = std::stoi(env("THREADS", "0"));
  s.maxBatch = std::stoi(env("MAX_BATCH", std::to_string(s.maxBatch)));
  s.maxTexts = std::stoul(env("MAX_TEXTS", std::to_string(s.maxTexts)));
  s.maxChars = std::stoul(env("MAX_CHARS", std::to_string(s.maxChars)));
  s.httpThreads = std::stoi(env("HTTP_THREADS", std::to_string(s.httpThreads)));
  s.pinFromCore = std::stoi(env("PIN_FROM_CORE", "-1"));
  s.idleSeconds = std::stoi(env("IDLE_SECONDS", std::to_string(s.idleSeconds)));
  s.preload = env("PRELOAD", "") == "1" || env("PRELOAD", "") == "true";
  s.token = env("TRANSLATE_TOKEN", "");
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (a == "--models") s.models = value();
    else if (a == "--host") s.host = value();
    else if (a == "--port") s.port = std::stoi(value());
    else if (a == "--threads") s.threads = std::stoi(value());
    else if (a == "--max-batch") s.maxBatch = std::stoi(value());
    else if (a == "--max-texts") s.maxTexts = std::stoul(value());
    else if (a == "--max-chars") s.maxChars = std::stoul(value());
    else if (a == "--http-threads") s.httpThreads = std::stoi(value());
    else if (a == "--pin-from-core") s.pinFromCore = std::stoi(value());
    else if (a == "--idle-seconds") s.idleSeconds = std::stoi(value());
    else if (a == "--preload") s.preload = true;
    else throw std::invalid_argument("unknown option " + a);
  }
  if (s.threads <= 0) s.threads = availableCpus();
  if (s.port <= 0 || s.port > 65535) throw std::invalid_argument("bad port");
  if (s.maxBatch < 1 || s.httpThreads < 1 || s.maxTexts < 1 || s.maxChars < 1)
    throw std::invalid_argument("MAX_BATCH, HTTP_THREADS, MAX_TEXTS and MAX_CHARS must be at least 1");
  return s;
}

void reply(httplib::Response &res, int status, const json &body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

void error(httplib::Response &res, int status, const std::string &message) { reply(res, status, {{"error", message}}); }

}  // namespace

int main(int argc, char **argv) {
  // SIGTERM/SIGINT are handled by one thread with sigwait (stopping the server from a signal
  // handler would not be async-signal-safe); block them before any other thread starts.
  sigset_t stopSignals;
  sigemptyset(&stopSignals);
  sigaddset(&stopSignals, SIGTERM);
  sigaddset(&stopSignals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &stopSignals, nullptr);

  Settings s;
  try {
    s = parse(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "reseam-translate: %s\n", e.what());
    return 2;
  }

  reseam::mt::EngineOptions opt;
  opt.modelsDir = s.models;
  opt.workers = s.threads;
  opt.maxBatch = s.maxBatch;
  opt.preload = s.preload;
  opt.firstCore = s.pinFromCore;
  opt.idleSeconds = s.idleSeconds;
  std::unique_ptr<reseam::mt::Engine> engine;
  try {
    engine = std::make_unique<reseam::mt::Engine>(opt);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "reseam-translate: %s\n", e.what());
    return 1;
  }

  httplib::Server server;
  server.new_task_queue = [&] { return new httplib::ThreadPool(size_t(s.httpThreads)); };
  // JSON may escape every character as \uXXXX (6 bytes per 2-byte character, 12 per emoji), so
  // allow 6x here; the decoded text is held to maxChars below.
  server.set_payload_max_length(s.maxChars * 6 + 4096);

  // Errors raised by the HTTP layer itself (body too large, unknown path) as JSON too.
  server.set_error_handler([](const httplib::Request &, httplib::Response &res) {
    if (!res.body.empty()) return;
    const char *message = res.status == 413 ? "Request body too large" : res.status == 404 ? "Not found" : "Request failed";
    res.set_content(json{{"error", message}}.dump(), "application/json");
  });

  server.set_pre_routing_handler([&](const httplib::Request &req, httplib::Response &res) {
    if (s.token.empty() || req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
    if (req.get_header_value("Authorization") != "Bearer " + s.token) {
      error(res, 401, "Missing or wrong token");
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  server.Get("/health", [&](const httplib::Request &, httplib::Response &res) {
    const auto st = engine->stats();
    reply(res, 200,
          {{"status", "ok"},
           {"isa", engine->isa()},
           {"threads", s.threads},
           {"pairs", engine->pairs().size()},
           {"sentences", st.sentences},
           {"tokens", st.tokens}});
  });

  server.Get("/v1/languages", [&](const httplib::Request &, httplib::Response &res) {
    json pairs = json::array();
    for (const auto &[from, to] : engine->pairs()) pairs.push_back({from, to});
    reply(res, 200, {{"pairs", pairs}, {"pivot", "en"}});
  });

  server.Post("/v1/translate", [&](const httplib::Request &req, httplib::Response &res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (const std::exception &) {
      return error(res, 400, "Body is not valid JSON");
    }
    if (!body.is_object() || !body.contains("from") || !body.contains("to") || !body.contains("texts") ||
        !body["from"].is_string() || !body["to"].is_string() || !body["texts"].is_array())
      return error(res, 400, "Expected {\"from\": string, \"to\": string, \"texts\": [string]}");
    const std::string from = body["from"], to = body["to"];
    if (from.empty() || from.size() > 16 || to.empty() || to.size() > 16) return error(res, 400, "Bad language code");
    const auto &arr = body["texts"];
    if (arr.size() > s.maxTexts) return error(res, 413, "Too many texts (at most " + std::to_string(s.maxTexts) + ")");
    std::vector<std::string> texts;
    size_t chars = 0;
    for (const auto &t : arr) {
      if (!t.is_string()) return error(res, 400, "Every text must be a string");
      texts.push_back(t.get<std::string>());
      chars += texts.back().size();
    }
    if (chars > s.maxChars) return error(res, 413, "Too much text (at most " + std::to_string(s.maxChars) + " bytes)");
    try {
      reply(res, 200, {{"translations", engine->translate(from, to, texts)}});
    } catch (const reseam::mt::InputError &e) {
      error(res, 400, e.what());
    } catch (const std::exception &e) {
      std::fprintf(stderr, "reseam-translate: %s-%s failed: %s\n", from.c_str(), to.c_str(), e.what());
      error(res, 500, "Translation failed");
    }
  });

  std::thread([&] {
    int sig;
    sigwait(&stopSignals, &sig);
    server.stop();
  }).detach();
  std::fprintf(stderr, "reseam-translate: %zu directions in %s, %d threads (%s), listening on %s:%d\n",
               engine->pairs().size(), s.models.c_str(), s.threads, engine->isa(), s.host.c_str(), s.port);
  if (!server.listen(s.host, s.port)) {
    std::fprintf(stderr, "reseam-translate: cannot listen on %s:%d\n", s.host.c_str(), s.port);
    return 1;
  }
  return 0;
}
