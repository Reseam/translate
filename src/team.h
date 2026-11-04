// A persistent team of threads that runs one job at a time, all threads together (SPMD).
//
// The decoder is weight-bandwidth bound: each step multiplies a few rows by ~7 MB of weights.
// Splitting every weight matrix by columns across the team means each core only ever touches its
// own slice, which stays in its private L2 from step to step, and a single sentence gets every
// core. Jobs are dispatched with a generation counter; between jobs the workers spin briefly
// (decode steps come back to back) and then sleep, so an idle engine uses no CPU.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace reseam::mt {

class Team {
 public:
  /// `size` threads including the caller of run(). With `firstCore` >= 0, thread i is pinned to
  /// core firstCore + i (so cores below firstCore stay free for I/O and request threads).
  explicit Team(int size, int firstCore = -1);
  ~Team();
  Team(const Team &) = delete;
  Team &operator=(const Team &) = delete;

  int size() const { return size_; }

  /// Runs fn(thread, size) on every thread (the caller is thread 0) and returns when all finish.
  /// fn must not throw: the other threads may be waiting for it at a barrier.
  template <class F>
  void run(F &&fn) {
    if (size_ == 1) return fn(0, 1);
    job_ = [&fn](int t, int n) { fn(t, n); };
    dispatch();
  }

  /// Inside run(): waits until every thread of the team reaches the barrier. Lets one job hold a
  /// whole multi-phase computation (a full decoder step) without re-dispatching per phase.
  void barrier() {
    if (size_ == 1) return;
    const uint64_t g = barrierGeneration_.load(std::memory_order_acquire);
    if (barrierCount_.fetch_add(1, std::memory_order_acq_rel) == size_ - 1) {
      barrierCount_.store(0, std::memory_order_relaxed);
      barrierGeneration_.fetch_add(1, std::memory_order_release);
    } else {
      while (barrierGeneration_.load(std::memory_order_acquire) == g) _mm_pause();
    }
  }

  /// [begin, end) of `n` items for thread t, in multiples of `align` (the last range takes the rest).
  static void split(int n, int t, int threads, int align, int &begin, int &end) {
    const int units = (n + align - 1) / align;
    const int b = int(int64_t(units) * t / threads), e = int(int64_t(units) * (t + 1) / threads);
    begin = std::min(n, b * align);
    end = std::min(n, e * align);
  }

 private:
  void dispatch();
  void worker(int index);

  int size_;
  std::vector<std::thread> threads_;
  std::function<void(int, int)> job_;
  alignas(64) std::atomic<uint64_t> generation_{0};
  alignas(64) std::atomic<int> pending_{0};
  alignas(64) std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  alignas(64) std::atomic<int> barrierCount_{0};
  alignas(64) std::atomic<uint64_t> barrierGeneration_{0};
};

}  // namespace reseam::mt
