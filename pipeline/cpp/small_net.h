// The small value network of arch/nnue_distill.py (70 -> H -> 32 -> 1,
// clipped ReLU, tanh) on the CPU, in float and quantized to 8 bits
// (worklog 2026-09-25-nn-architectures, entries 21-23).
//
// Inputs are the uint8 network states (the 70 inputs x 4: board bits 0/4,
// reserve counts 0-4), as Game::writeGameState writes them x 4.
//
// int8: layer 1 accumulates int32 (weights x 254, i.e. activations scaled
// 127 with 3 fractional bits), clipped to [0, 127] as uint8. Layer 2 is
// uint8 x int8 -> int32 with AVX-VNNI (vpdpbusd), the weights laid out
// [H/4][32][4] so that 4 activations at a time update all 32 outputs and
// groups of 4 zero activations are skipped. Layer 3 (32 -> 1) is float.
#ifndef PIPELINE_SMALL_NET_H
#define PIPELINE_SMALL_NET_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

class SmallNet {
 public:
  static constexpr int32_t kMaxHidden = 2048;
  explicit SmallNet(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr || std::fread(&h_, 4, 1, f) != 1 || h_ % 32 != 0 ||
        h_ > kMaxHidden) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      std::exit(1);
    }
    auto read = [&](std::vector<float> &v, size_t n) {
      v.resize(n);
      if (std::fread(v.data(), 4, n, f) != n)
        std::exit(1);
    };
    std::vector<float> w2_rows;
    read(w1_, 70 * static_cast<size_t>(h_));  // input-major [70][h]
    read(b1_, h_);
    read(w2_rows, 32 * static_cast<size_t>(h_));  // [32][h]
    read(b2_, 32);
    read(w3_, 32);
    if (std::fread(&b3_, 4, 1, f) != 1)
      std::exit(1);
    std::fclose(f);
    // Float layer 2, input-major [h][32]
    w2_.resize(w2_rows.size());
    for (int32_t k = 0; k < 32; ++k)
      for (int32_t j = 0; j < h_; ++j)
        w2_[static_cast<size_t>(j) * 32 + k] =
            w2_rows[static_cast<size_t>(k) * h_ + j];
    quantize(w2_rows);
  }

  int32_t hidden() const noexcept { return h_; }

  /// Value for the side to move, float
  float evalFloat(const uint8_t *state) const {
    // Stack arrays and restrict pointers: with thread_local vectors the
    // compiler could not rule out aliasing and did not vectorize
    alignas(32) float a[kMaxHidden];
    float *__restrict ap = a;
    const int32_t h = h_;
    std::copy(b1_.begin(), b1_.end(), ap);
    for (int32_t i = 0; i < 70; ++i) {
      if (state[i] == 0)
        continue;
      const float xi = 0.25F * static_cast<float>(state[i]);
      const float *__restrict w = w1_.data() + static_cast<size_t>(i) * h;
      for (int32_t j = 0; j < h; ++j)
        ap[j] += xi * w[j];
    }
    alignas(32) float h2[32];
    float *__restrict hp = h2;
    std::copy(b2_.begin(), b2_.end(), hp);
    for (int32_t j = 0; j < h; ++j) {
      const float aj = std::min(std::max(ap[j], 0.0F), 1.0F);
      if (aj == 0.0F)
        continue;
      const float *__restrict w = w2_.data() + static_cast<size_t>(j) * 32;
      for (int32_t k = 0; k < 32; ++k)
        hp[k] += aj * w[k];
    }
    return output(h2, 1.0F);
  }

  /// Value for the side to move, 8-bit
  float evalInt8(const uint8_t *state) const {
    alignas(32) int32_t acc_buf[kMaxHidden];
    alignas(32) uint8_t act_buf[kMaxHidden];
    int32_t *__restrict acc = acc_buf;
    uint8_t *__restrict act = act_buf;
    const int32_t h = h_;
    std::copy(b1q_.begin(), b1q_.end(), acc);
    for (int32_t i = 0; i < 70; ++i) {
      const int32_t x = state[i];
      if (x == 0)
        continue;
      const int16_t *__restrict w = w1q_.data() + static_cast<size_t>(i) * h;
      for (int32_t j = 0; j < h; ++j)
        acc[j] += x * w[j];
    }
    return fromAccumulator(acc);
  }

  /// Incremental evaluation (entry 23). A move changes only a few of the
  /// 64 board inputs (the bits of the board word), so layer 1's board part
  /// is kept per position and updated by the changed bits; the 6 reserve
  /// inputs (counts 0-4, the side to move's first) are added at evaluation
  /// from precomputed rows. Same integers as evalInt8, so the same result.
  /// @param acc h int32: the bias plus the board bits' rows
  void boardAccumulator(uint64_t board, int32_t *acc) const {
    std::copy(b1q_.begin(), b1q_.end(), acc);
    for (; board != 0; board &= board - 1)
      addRow(acc, board_rows_.data() +
                      static_cast<size_t>(__builtin_ctzll(board)) * h_);
  }
  /// The child's board accumulator from its parent's
  void updateAccumulator(const int32_t *parent, uint64_t parent_board,
                         uint64_t child_board, int32_t *child) const {
    std::copy(parent, parent + h_, child);
    for (uint64_t on = child_board & ~parent_board; on != 0; on &= on - 1)
      addRow(child, board_rows_.data() +
                        static_cast<size_t>(__builtin_ctzll(on)) * h_);
    for (uint64_t off = parent_board & ~child_board; off != 0; off &= off - 1)
      subRow(child, board_rows_.data() +
                        static_cast<size_t>(__builtin_ctzll(off)) * h_);
  }
  /// Value for the side to move from the parent's board accumulator, in
  /// one pass (the child's own accumulator is never stored: lazy, entry 23)
  float evalFromParent(const int32_t *parent, uint64_t parent_board,
                       uint64_t board, const int32_t reserves[6]) const {
    alignas(32) int32_t acc_buf[kMaxHidden];
    int32_t *__restrict acc = acc_buf;
    std::copy(parent, parent + h_, acc);
    for (uint64_t on = board & ~parent_board; on != 0; on &= on - 1)
      addRow(acc, board_rows_.data() +
                      static_cast<size_t>(__builtin_ctzll(on)) * h_);
    for (uint64_t off = parent_board & ~board; off != 0; off &= off - 1)
      subRow(acc, board_rows_.data() +
                      static_cast<size_t>(__builtin_ctzll(off)) * h_);
    for (int32_t i = 0; i < 6; ++i)
      if (reserves[i] != 0)
        addRow(acc, reserve_rows_.data() +
                        (static_cast<size_t>(i) * 5 + reserves[i]) * h_);
    return fromAccumulator(acc);
  }
  /// Value for the side to move from a board accumulator and the reserve
  /// counts (side to move's three first, as the network inputs)
  float evalIncremental(const int32_t *board_acc,
                        const int32_t reserves[6]) const {
    alignas(32) int32_t acc_buf[kMaxHidden];
    int32_t *__restrict acc = acc_buf;
    std::copy(board_acc, board_acc + h_, acc);
    for (int32_t i = 0; i < 6; ++i)
      if (reserves[i] != 0)
        addRow(acc, reserve_rows_.data() +
                        (static_cast<size_t>(i) * 5 + reserves[i]) * h_);
    return fromAccumulator(acc);
  }

 private:
  void addRow(int32_t *__restrict acc, const int32_t *__restrict row) const {
    for (int32_t j = 0; j < h_; ++j)
      acc[j] += row[j];
  }
  void subRow(int32_t *__restrict acc, const int32_t *__restrict row) const {
    for (int32_t j = 0; j < h_; ++j)
      acc[j] -= row[j];
  }

  /// Layers 2 and 3 from layer 1's accumulator
  float fromAccumulator(const int32_t *__restrict acc) const {
    alignas(32) uint8_t act_buf[kMaxHidden];
    uint8_t *__restrict act = act_buf;
    const int32_t h = h_;
    for (int32_t j = 0; j < h; ++j)
      act[j] = static_cast<uint8_t>(std::min(std::max(acc[j] >> 3, 0), 127));
    __m256i sum[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                      _mm256_setzero_si256(), _mm256_setzero_si256()};
    const uint32_t *groups = reinterpret_cast<const uint32_t *>(act_buf);
    for (int32_t g = 0; g < h_ / 4; ++g) {
      if (groups[g] == 0)
        continue;  // four zero activations
      const __m256i in = _mm256_set1_epi32(static_cast<int32_t>(groups[g]));
      const __m256i *w =
          reinterpret_cast<const __m256i *>(w2q_.data() + g * 128);
      for (int32_t r = 0; r < 4; ++r)
        sum[r] =
            _mm256_dpbusd_avx_epi32(sum[r], in, _mm256_loadu_si256(w + r));
    }
    alignas(32) int32_t out[32];
    for (int32_t r = 0; r < 4; ++r)
      _mm256_store_si256(reinterpret_cast<__m256i *>(out + 8 * r), sum[r]);
    float h2[32];
    for (int32_t k = 0; k < 32; ++k)
      h2[k] = static_cast<float>(out[k]) + b2q_[k];
    return output(h2, 1.0F / (127.0F * w2_scale_));
  }

  float output(const float *h2, float scale) const {
    float out = b3_;
    for (int32_t k = 0; k < 32; ++k)
      out += std::min(std::max(h2[k] * scale, 0.0F), 1.0F) * w3_[k];
    return std::tanh(out);
  }

  void quantize(const std::vector<float> &w2_rows) {
    // Layer 1: accumulator units of 1/(127 * 8); inputs are x 4
    w1q_.resize(w1_.size());
    for (size_t i = 0; i < w1_.size(); ++i)
      w1q_[i] = static_cast<int16_t>(std::lround(w1_[i] * 254.0F));
    b1q_.resize(h_);
    for (int32_t j = 0; j < h_; ++j)
      b1q_[j] = static_cast<int32_t>(std::lround(b1_[j] * 1016.0F));
    // Incremental rows: a board bit is input 4; reserve slot i with count c
    board_rows_.resize(64 * static_cast<size_t>(h_));
    for (int32_t b = 0; b < 64; ++b)
      for (int32_t j = 0; j < h_; ++j)
        board_rows_[static_cast<size_t>(b) * h_ + j] =
            4 * w1q_[static_cast<size_t>(b) * h_ + j];
    reserve_rows_.resize(6 * 5 * static_cast<size_t>(h_));
    for (int32_t i = 0; i < 6; ++i)
      for (int32_t c = 0; c < 5; ++c)
        for (int32_t j = 0; j < h_; ++j)
          reserve_rows_[(static_cast<size_t>(i) * 5 + c) * h_ + j] =
              c * w1q_[static_cast<size_t>(64 + i) * h_ + j];
    // Layer 2: weights x 64, saturated at +-127 (|w| <= 1.98, as NNUE
    // clips them in training); SMALL_NET_W2_SCALE overrides the scale
    const char *env = std::getenv("SMALL_NET_W2_SCALE");
    w2_scale_ = env != nullptr ? static_cast<float>(std::atof(env)) : 64.0F;
    w2q_.resize(w2_rows.size());
    for (int32_t g = 0; g < h_ / 4; ++g)
      for (int32_t k = 0; k < 32; ++k)
        for (int32_t b = 0; b < 4; ++b)
          w2q_[static_cast<size_t>(g) * 128 + k * 4 + b] =
              static_cast<int8_t>(std::clamp<long>(
                  std::lround(
                      w2_rows[static_cast<size_t>(k) * h_ + 4 * g + b] *
                      w2_scale_),
                  -127, 127));
    for (int32_t k = 0; k < 32; ++k)
      b2q_[k] = b2_[k] * 127.0F * w2_scale_;
  }

  int32_t h_{0};
  std::vector<float> w1_, b1_, w2_, b2_, w3_;
  float b3_{0};
  std::vector<int16_t> w1q_;
  std::vector<int32_t> b1q_;
  std::vector<int32_t> board_rows_, reserve_rows_;
  std::vector<int8_t> w2q_;
  float b2q_[32]{};
  float w2_scale_{1};
};

#endif
