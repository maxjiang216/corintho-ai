#ifndef ARENA_H
#define ARENA_H

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <memory>
#include <mutex>
#include <vector>

/// @brief A bump allocator with size-class free lists, for search tree nodes
/// @details The search churns through far more nodes than it ever holds: at
/// 1600 searches a game peaks at about 3,300 live blocks while allocating
/// 527,076 over ten games, because moveDown frees a whole subtree every move
/// and the next turn rebuilds one. Handing that to malloc put 34% of all L1
/// data read misses inside malloc_consolidate and unlink_chunk -- glibc
/// walking its own free-list metadata, work unrelated to our data.
///
/// Blocks are never returned to the OS. Freed blocks go on a size-class list
/// and are handed straight back out, so steady-state search does no system
/// allocation at all.
///
/// Each thread has an instance, and blocks move between threads in batches.
/// A block is freed on whichever thread runs its game at the time, and since
/// the dynamic OpenMP schedule (worklog entry 19) that is often not the thread
/// that allocated it. With purely per-thread free lists, blocks piled up in
/// some threads while others kept carving new memory: resident memory grew by
/// ~290 MB per 1,000 games played at 20 threads and never came back (entry
/// 29). Now a thread keeps at most two batches of free blocks per size class
/// and passes surplus batches to a shared pool, which a thread with none takes
/// from before carving new memory. The pool's mutex is taken once per kBatch
/// frees or allocations at most.
class Arena {
 public:
  Arena() {
    // spill() never holds more than kLocalBatches + 1, so the per-thread
    // lists never reallocate and deallocate() cannot throw from them
    for (auto &v : spare_)
      v.reserve(kLocalBatches + 1);
  }
  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  /// @brief Size classes are powers of two from 64 to 1024 bytes. Node is 64;
  /// an edge block is 128; a node's child-statistics block is 128 to 1024
  /// depending on how many children it holds room for.
  static constexpr size_t kSmall = 64;
  static constexpr size_t kLarge = 128;
  static constexpr size_t kMaxBlock = 1024;

  /// @warning A block must be returned to the class it came from. Callers
  /// that round a request up to a class must round the matching deallocation
  /// up the same way, or the free lists will be crossed.
  void *allocate(size_t bytes) {
    const int c = sizeClass(bytes);
    if (count_[c] == 0 && !refill(c)) {
      const size_t slot = kSmall << c;
      if (left_ < slot)
        grow();
      void *p = cur_;
      cur_ += slot;
      left_ -= slot;
      return p;
    }
    void *p = free_[c];
    free_[c] = *static_cast<void **>(p);  // next pointer lives in the block
    --count_[c];
    return p;
  }

  void deallocate(void *p, size_t bytes) noexcept {
    if (p == nullptr)
      return;
    const int c = sizeClass(bytes);
    if (count_[c] == kBatch)
      spill(c);
    *static_cast<void **>(p) = free_[c];
    free_[c] = p;
    ++count_[c];
  }

  /// @brief The per-thread instance
  static Arena &get() noexcept {
    static thread_local Arena arena;
    return arena;
  }

 private:
  static constexpr int kNumClasses = 5;
  /// @brief Blocks per batch passed between a thread and the pool
  static constexpr int32_t kBatch = 256;
  /// @brief Full batches a thread keeps per class before passing surplus to
  /// the pool. Keeping only one cost 11% engine time at 20 threads: blocks a
  /// thread had just freed, still in its core's cache, went to other threads
  /// cold (entry 29).
#ifndef ARENA_LOCAL_BATCHES
#define ARENA_LOCAL_BATCHES 16
#endif
  static constexpr size_t kLocalBatches = ARENA_LOCAL_BATCHES;
  /// @brief 64 KiB at a time: about 1000 nodes, a third of a game's peak
  static constexpr size_t kChunk = 1U << 16;

  /// @brief Shared by all threads: full batches of free blocks, and every
  /// chunk ever carved. Chunks are owned here, not by a thread, because their
  /// blocks end up on every thread's lists.
  struct Pool {
    std::mutex mutex;
    std::vector<void *> batches[kNumClasses];  // heads of kBatch-block lists
    /// @brief batches[c].size(), readable without the mutex. A thread with
    /// no free blocks checks it first: taking the mutex just to find the pool
    /// empty, on every allocation while carving, cost ~5% at 20 threads.
    std::atomic<int32_t> available[kNumClasses]{};
    std::vector<std::unique_ptr<unsigned char[]>> chunks;
  };
  /// @brief Never destroyed: blocks may still be in use by static objects'
  /// destructors at exit, and the OS reclaims the memory then anyway
  static Pool &pool() {
    static Pool *p = new Pool;
    return *p;
  }

  /// @brief The current list is full: keep it as a local full batch, and
  /// hand the oldest local batches to the pool if there are too many
  /// @details Keeping several batches gives hysteresis (a thread that frees
  /// and allocates alternately around a batch boundary does not take the
  /// mutex every time) and keeps recently freed blocks on this core.
  void spill(int c) noexcept {
    std::vector<void *> &mine = spare_[c];
    mine.push_back(free_[c]);
    free_[c] = nullptr;
    count_[c] = 0;
    if (mine.size() > kLocalBatches) {
      // Pass the older half: the newest batches are the likeliest in cache
      const size_t keep = kLocalBatches / 2;
      const size_t give = mine.size() - keep;
      Pool &p = pool();
      std::lock_guard<std::mutex> lock{p.mutex};
      p.batches[c].insert(p.batches[c].end(), mine.begin(), mine.begin() + give);
      p.available[c].store(static_cast<int32_t>(p.batches[c].size()),
                           std::memory_order_relaxed);
      mine.erase(mine.begin(), mine.begin() + give);
    }
  }

  /// @brief The current list is empty: take a local batch, or one from the
  /// pool. False if neither has one, and the caller carves a new block.
  bool refill(int c) {
    std::vector<void *> &mine = spare_[c];
    if (mine.empty()) {
      Pool &p = pool();
      // A stale read only means carving one block more, or taking the mutex
      // and finding the pool empty after all; both are safe
      if (p.available[c].load(std::memory_order_relaxed) == 0)
        return false;
      std::lock_guard<std::mutex> lock{p.mutex};
      if (p.batches[c].empty())
        return false;
      mine.push_back(p.batches[c].back());
      p.batches[c].pop_back();
      p.available[c].store(static_cast<int32_t>(p.batches[c].size()),
                           std::memory_order_relaxed);
    }
    free_[c] = mine.back();
    mine.pop_back();
    count_[c] = kBatch;
    return true;
  }

  void grow() {
    // A leftover tail smaller than the request is abandoned. At most 1 KiB of
    // each 64 KiB chunk, and only when a large block straddles the end.
    auto chunk = std::make_unique<unsigned char[]>(kChunk);
    cur_ = chunk.get();
    left_ = kChunk;
    Pool &p = pool();
    std::lock_guard<std::mutex> lock{p.mutex};
    p.chunks.push_back(std::move(chunk));
  }

  /// @brief 0 for up to 64 bytes, 1 for up to 128, ... 4 for up to 1024
  static int sizeClass(size_t bytes) noexcept {
    if (bytes <= kSmall)
      return 0;
    // Bit length of bytes - 1, minus 6: 65..128 -> 1, 129..256 -> 2, ...
    return 64 - __builtin_clzll(static_cast<unsigned long long>(bytes - 1)) - 6;
  }

  unsigned char *cur_{nullptr};
  size_t left_{0};
  /// @brief Current free list per class, count_ blocks (at most kBatch)
  void *free_[kNumClasses]{};
  int32_t count_[kNumClasses]{};
  /// @brief Full batches (kBatch blocks each) kept by this thread, per class,
  /// newest last
  std::vector<void *> spare_[kNumClasses];
};

#endif
