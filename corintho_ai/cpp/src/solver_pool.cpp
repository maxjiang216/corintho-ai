#include "solver_pool.h"

#include <chrono>

#include "solver.h"

SolverPool::SolverPool(int32_t num_threads, int32_t log2_table,
                       uint64_t max_nodes)
    : log2_table_{log2_table}, max_nodes_{max_nodes} {
  for (int32_t i = 0; i < num_threads; ++i)
    threads_.emplace_back([this, i] { run(i); });
}

SolverPool::~SolverPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread &t : threads_)
    t.join();
}

std::shared_ptr<SolveJob> SolverPool::submit(const Game &game) {
  auto job = std::make_shared<SolveJob>();
  job->game = game;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(job);
  }
  wake_.notify_one();
  return job;
}

double SolverPool::seconds() const noexcept {
  return static_cast<double>(nanos_.load()) * 1e-9;
}

double SolverPool::max_seconds() const noexcept {
  return static_cast<double>(max_nanos_.load()) * 1e-9;
}

void SolverPool::run(int32_t index) {
  (void)index;
  Solver solver{log2_table_};  // never cleared: see solver_pool.h
  for (;;) {
    std::shared_ptr<SolveJob> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty())
        return;
      job = std::move(queue_.front());
      queue_.pop_front();
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int32_t result = solver.solve(job->game, max_nodes_);
    const auto ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
    nanos_ += ns;
    uint64_t prev = max_nanos_.load();
    while (ns > prev && !max_nanos_.compare_exchange_weak(prev, ns)) {
    }
    ++solves_;
    if (result == Solver::kUnknown)
      ++capped_;
    job->result.store(static_cast<int8_t>(result), std::memory_order_release);
  }
}
