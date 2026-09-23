#ifndef ARENA_H
#define ARENA_H

#include <cstddef>
#include <cstdint>

#include <memory>
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
/// @note One instance per thread. Trees do not migrate between threads: each
/// game owns its own trees, and a tree is only ever touched by the thread
/// running that game.
class Arena {
 public:
  Arena() = default;
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
    void *&list = free_[c];
    if (list != nullptr) {
      void *p = list;
      list = *static_cast<void **>(p);  // next pointer lives in the free block
      return p;
    }
    const size_t slot = kSmall << c;
    if (left_ < slot)
      grow();
    void *p = cur_;
    cur_ += slot;
    left_ -= slot;
    return p;
  }

  void deallocate(void *p, size_t bytes) noexcept {
    if (p == nullptr)
      return;
    void *&list = free_[sizeClass(bytes)];
    *static_cast<void **>(p) = list;
    list = p;
  }

  /// @brief The per-thread instance
  static Arena &get() noexcept {
    static thread_local Arena arena;
    return arena;
  }

 private:
  void grow() {
    // A leftover tail smaller than the request is abandoned. At most 1 KiB of
    // each 64 KiB chunk, and only when a large block straddles the end.
    chunks_.push_back(std::make_unique<unsigned char[]>(kChunk));
    cur_ = chunks_.back().get();
    left_ = kChunk;
  }

  /// @brief 64 KiB at a time: about 1000 nodes, a third of a game's peak
  static constexpr size_t kChunk = 1U << 16;
  std::vector<std::unique_ptr<unsigned char[]>> chunks_;
  unsigned char *cur_{nullptr};
  size_t left_{0};
  static constexpr int kNumClasses = 5;
  /// @brief 0 for up to 64 bytes, 1 for up to 128, ... 4 for up to 1024
  static int sizeClass(size_t bytes) noexcept {
    if (bytes <= kSmall)
      return 0;
    // Bit length of bytes - 1, minus 6: 65..128 -> 1, 129..256 -> 2, ...
    return 64 - __builtin_clzll(static_cast<unsigned long long>(bytes - 1)) - 6;
  }
  void *free_[kNumClasses]{};
};

#endif
