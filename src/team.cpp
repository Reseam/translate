#include "team.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <chrono>

namespace reseam::mt {

namespace {

// How long a worker spins for the next job before sleeping. Decode steps dispatch jobs every
// few microseconds, so this keeps the team hot while busy without burning an idle core for long.
constexpr auto kSpin = std::chrono::microseconds(200);

void pin(std::thread &t, int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
}

}  // namespace

Team::Team(int size, int firstCore) : size_(std::max(1, size)) {
  for (int i = 1; i < size_; ++i) {
    threads_.emplace_back([this, i] { worker(i); });
    if (firstCore >= 0) pin(threads_.back(), firstCore + i);
  }
  if (firstCore >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(firstCore, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
}

Team::~Team() {
  stop_.store(true);
  generation_.fetch_add(1, std::memory_order_release);
  generation_.notify_all();
  for (auto &t : threads_) t.join();
}

void Team::dispatch() {
  pending_.store(size_ - 1, std::memory_order_relaxed);
  // seq_cst on both sides of the sleep handshake (generation_ then sleepers_ here, sleepers_
  // then generation_ in worker): with acquire/release alone both could read stale values, the
  // worker would sleep through this job and dispatch would wait for it forever.
  generation_.fetch_add(1, std::memory_order_seq_cst);
  if (sleepers_.load(std::memory_order_seq_cst) > 0) generation_.notify_all();
  job_(0, size_);
  // Wait for the others; their slices are about as long as ours, so spin.
  while (pending_.load(std::memory_order_acquire) != 0) _mm_pause();
}

void Team::worker(int index) {
  uint64_t seen = 0;
  for (;;) {
    uint64_t g = generation_.load(std::memory_order_acquire);
    if (g == seen) {
      const auto until = std::chrono::steady_clock::now() + kSpin;
      int spins = 0;
      while ((g = generation_.load(std::memory_order_acquire)) == seen) {
        _mm_pause();
        if (++spins % 64 == 0 && std::chrono::steady_clock::now() > until) {
          sleepers_.fetch_add(1, std::memory_order_seq_cst);
          generation_.wait(seen, std::memory_order_seq_cst);
          sleepers_.fetch_sub(1, std::memory_order_seq_cst);
          spins = 0;
        }
      }
    }
    seen = g;
    if (stop_.load(std::memory_order_acquire)) return;
    job_(index, size_);
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

}  // namespace reseam::mt
