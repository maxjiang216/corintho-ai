// Network evaluation cache for self-play (worklog 2026-09-25-nn-architectures,
// entry 07).
//
// In self-play half of all network rows repeat a position evaluated earlier in
// the run (a third within the same game), and 56% do once positions are
// identified up to the 8 board symmetries. This cache stores one value and
// fp16 policy per position, under an exact key: the 16 spaces' 4-bit contents
// in the symmetric frame whose packed board is least, plus the 6 reserve
// counts. The policy is stored in that canonical frame and rotated back into
// each row's own frame on a hit.
//
// A hit from another frame returns the network's evaluation of a symmetric
// copy, which differs from the row's own (the network is only approximately
// invariant: value correlation ~0.97); AlphaGo Zero evaluated every position
// in a random frame on purpose. Hits from the same frame are exact. With
// symmetric = false only exact positions match.
//
// 4-way set associative, replacing the oldest entry of a set. A set's four
// keys and ages fill one 64-byte line, so a lookup reads one line (prefetched
// a batch ahead) and a hit one more payload. Single-threaded: the driver runs
// every network call, cache included, on one worker thread, off the engine's
// critical path.

#ifndef PIPELINE_EVAL_CACHE_H
#define PIPELINE_EVAL_CACHE_H

#include <immintrin.h>

#include <cstdint>

#include <algorithm>
#include <vector>

#include "util.h"

class EvalCache {
 public:
  // The rows of one network call: keys, frames, sets, and which rows missed
  struct Call {
    std::vector<uint64_t> board, set;
    std::vector<uint32_t> reserves;
    std::vector<int8_t> frame;
    std::vector<int8_t> stored_frame;  // of each hit's entry (diagnostics)
    std::vector<int32_t> misses;
    std::vector<int64_t> slot;  // payload index of each row: hit, or victim
  };

  // 2^log2_entries entries in sets of kWays
  explicit EvalCache(int32_t log2_entries, bool symmetric = true)
      : keys_(size_t{1} << std::max(0, log2_entries - 2)),
        payload_(keys_.size() * kWays),
        mask_(keys_.size() - 1),
        frames_(symmetric ? kNumSymmetries : 1) {
    for (int32_t k = 0; k < kNumSymmetries; ++k)
      for (int32_t j = 0; j < kBoardSize; ++j)
        shuffle_[k][j] = static_cast<int8_t>(space_symmetries[k][j]);
  }

  // Fills values/probs for the rows found in the cache and lists the others
  // in call.misses, in row order
  void lookup(const float *states, int32_t rows, float *values, float *probs,
              Call &call) {
    const size_t n = static_cast<size_t>(rows);
    call.board.resize(n);
    call.reserves.resize(n);
    call.set.resize(n);
    call.frame.resize(n);
    call.stored_frame.assign(n, -1);
    call.misses.clear();
    // Keys first, prefetching each row's key line, then the probes
    for (size_t i = 0; i < n; ++i) {
      keyOf(states + i * kGameStateSize, call.board[i], call.reserves[i],
            call.frame[i]);
      call.set[i] = setOf(call.board[i], call.reserves[i]);
      _mm_prefetch(reinterpret_cast<const char *>(&keys_[call.set[i]]),
                   _MM_HINT_T0);
    }
    // Probe the key lines; prefetch each hit's payload (4 lines)
    call.slot.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const KeySet &ks = keys_[call.set[i]];
      int32_t way = -1;
      for (int32_t w = 0; w < kWays; ++w)
        if (ks.board[w] == call.board[i] && ks.reserves[w] == call.reserves[i])
          way = w;
      if (way < 0) {
        call.slot[i] = -1;
        call.misses.push_back(static_cast<int32_t>(i));
        continue;
      }
      call.slot[i] = static_cast<int64_t>(call.set[i] * kWays) + way;
      prefetchPayload(call.slot[i]);
    }
    for (size_t i = 0; i < n; ++i) {
      if (call.slot[i] < 0)
        continue;
      const Payload &e = payload_[static_cast<size_t>(call.slot[i])];
      call.stored_frame[i] = e.frame;
      values[i] = e.value;
      float canonical[kNumMoves];
      for (int32_t j = 0; j < kNumMoves; j += 8)
        _mm256_storeu_ps(canonical + j,
                         _mm256_cvtph_ps(_mm_loadu_si128(
                             reinterpret_cast<const __m128i *>(e.policy + j))));
      // The canonical policy is copy `frame` of the row's: canonical[j] =
      // policy[move_symmetries[frame][j]] (as SelfPlayer::writeSamples)
      float *p = probs + i * kNumMoves;
      const int32_t f = call.frame[i];
      for (int32_t j = 0; j < kNumMoves; ++j)
        p[move_symmetries[f][j]] = canonical[j];
    }
    lookups_ += n;
    hits_ += n - call.misses.size();
  }

  // Stores the evaluated misses; values/probs are in the call's row layout
  void insert(Call &call, const float *values, const float *probs) {
    const uint32_t now = ++clock_;
    // Claim a way for each miss (the key lines are still cached from the
    // lookup) and prefetch its payload; then write the payloads
    for (const int32_t r : call.misses) {
      const size_t i = static_cast<size_t>(r);
      KeySet &ks = keys_[call.set[i]];
      int32_t victim = 0;
      bool present = false;
      for (int32_t w = 0; w < kWays; ++w) {
        if (ks.board[w] == call.board[i] && ks.reserves[w] == call.reserves[i]) {
          present = true;  // the same position twice in one call
          break;
        }
        if (ks.stamp[w] < ks.stamp[victim])
          victim = w;
      }
      if (present) {
        call.slot[i] = -1;
        continue;
      }
      ks.board[victim] = call.board[i];
      ks.reserves[victim] = call.reserves[i];
      ks.stamp[victim] = now;
      call.slot[i] = static_cast<int64_t>(call.set[i] * kWays) + victim;
      prefetchPayload(call.slot[i]);
    }
    for (const int32_t r : call.misses) {
      const size_t i = static_cast<size_t>(r);
      if (call.slot[i] < 0)
        continue;
      Payload &e = payload_[static_cast<size_t>(call.slot[i])];
      const int32_t f = call.frame[i];
      const float *p = probs + i * kNumMoves;
      float canonical[kNumMoves];
      for (int32_t j = 0; j < kNumMoves; ++j)
        canonical[j] = p[move_symmetries[f][j]];
      e.value = values[r];
      e.frame = static_cast<int8_t>(f);
      for (int32_t j = 0; j < kNumMoves; j += 8)
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(e.policy + j),
            _mm256_cvtps_ph(_mm256_loadu_ps(canonical + j),
                            _MM_FROUND_TO_NEAREST_INT));
    }
  }

  uint64_t lookups() const { return lookups_; }
  uint64_t hits() const { return hits_; }

 private:
  static constexpr int32_t kWays = 4;
  static constexpr uint32_t kEmpty = ~uint32_t{0};  // no real reserves value
  struct alignas(64) KeySet {
    uint64_t board[kWays]{};
    uint32_t reserves[kWays]{kEmpty, kEmpty, kEmpty, kEmpty};
    uint32_t stamp[kWays]{};  // insertion clock; the oldest way is replaced
  };
  static_assert(sizeof(KeySet) == 64, "a set's keys should fill one line");
  struct Payload {
    float value{0.0F};
    int8_t frame{0};  // the inserting row's frame (diagnostics)
    uint16_t policy[kNumMoves]{};
  };

  // The exact key: the board as 16 bytes (space j: its 4 input bits), in
  // each symmetric copy's frame (copy k's space j is space
  // space_symmetries[k][j], as in SelfPlayer::writeSamples); the least copy,
  // packed 4 bits per space; and the 6 reserve counts, 4 bits each
  void keyOf(const float *row, uint64_t &board, uint32_t &reserves,
             int8_t &frame) const {
    alignas(16) uint8_t bytes[16];
    for (int32_t s = 0; s < kBoardSize; ++s) {
      const float *b = row + 4 * s;
      bytes[s] = static_cast<uint8_t>(
          (b[0] != 0.0F) | (b[1] != 0.0F) << 1 | (b[2] != 0.0F) << 2 |
          (b[3] != 0.0F) << 3);
    }
    const __m128i v = _mm_load_si128(reinterpret_cast<const __m128i *>(bytes));
    uint64_t best_hi = ~uint64_t{0}, best_lo = ~uint64_t{0};
    frame = 0;
    for (int32_t k = 0; k < frames_; ++k) {
      const __m128i c = _mm_shuffle_epi8(
          v, _mm_loadu_si128(reinterpret_cast<const __m128i *>(shuffle_[k])));
      const uint64_t lo = static_cast<uint64_t>(_mm_cvtsi128_si64(c));
      const uint64_t hi =
          static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(c, c)));
      if (hi < best_hi || (hi == best_hi && lo < best_lo)) {
        best_hi = hi;
        best_lo = lo;
        frame = static_cast<int8_t>(k);
      }
    }
    // Each byte is below 16: pack two per byte
    const uint64_t mask = 0x0F0F0F0F0F0F0F0FULL;
    board = _pext_u64(best_lo, mask) | _pext_u64(best_hi, mask) << 32;
    reserves = 0;
    for (int32_t i = 0; i < 6; ++i)
      reserves |= static_cast<uint32_t>(row[4 * kBoardSize + i] * 4.0F + 0.5F)
                  << (4 * i);
  }

  void prefetchPayload(int64_t slot) const {
    const char *p = reinterpret_cast<const char *>(&payload_[static_cast<size_t>(slot)]);
    for (size_t b = 0; b < sizeof(Payload); b += 64)
      _mm_prefetch(p + b, _MM_HINT_T0);
  }

  uint64_t setOf(uint64_t board, uint32_t reserves) const {
    uint64_t x = board * 0x9E3779B97F4A7C15ULL ^ (reserves + 0x632BE59BD9B4E019ULL);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return x & mask_;
  }

  std::vector<KeySet> keys_;
  std::vector<Payload> payload_;
  uint64_t mask_;
  int32_t frames_;
  alignas(16) int8_t shuffle_[kNumSymmetries][16];
  uint32_t clock_{0};
  uint64_t lookups_{0};
  uint64_t hits_{0};
};

#endif
