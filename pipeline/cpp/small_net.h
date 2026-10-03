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
  /// Incremental accumulators are 16-bit: the layer-1 sums of every
  /// position measured fit with room to spare (max 11896 of 32767), and
  /// wrap-around in intermediate sums cancels, so the results are the same
  /// integers as the 32-bit full evaluation, twice as many per instruction
  /// (entry 23)
  using Acc = int16_t;
  explicit SmallNet(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr || std::fread(&h_, 4, 1, f) != 1 || h_ % 64 != 0 ||
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

  /// Before the tanh, float
  float rawFloat(const uint8_t *state) const {
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
  /// Value for the side to move, float
  float evalFloat(const uint8_t *state) const {
    return std::tanh(rawFloat(state));
  }

  /// Value for the side to move, 8-bit
  float evalInt8(const uint8_t *state) const {
    return std::tanh(rawInt8(state));
  }
  /// A search score: the value before the tanh, / 8. The tanh only
  /// reshapes values, and alpha-beta only compares them, so the search
  /// skips it (entry 23); / 8 keeps every score within +-0.9 (the raw output
  /// spans about -4.7..7.2), below the win and loss scores.
  static constexpr float kScoreScale = 0.125F;
  float scoreInt8(const uint8_t *state) const {
    return kScoreScale * rawInt8(state);
  }
  float scoreFloat(const uint8_t *state) const {
    return kScoreScale * rawFloat(state);
  }
  /// Before the tanh, 8-bit
  float rawInt8(const uint8_t *state) const {
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
  /// Accumulators of a position (Game::key's board and rest words): two
  /// views, [t * h, t * h + h) for side t to move, each with layer 1's bias,
  /// board rows and reserve rows (the reserve inputs list the side to
  /// move's three counts first, so the views differ only there)
  void rootAccumulators(uint64_t board, uint64_t rest, Acc *acc) const {
    for (int32_t t = 0; t < 2; ++t) {
      Acc *view = acc + static_cast<size_t>(t) * h_;
      std::copy(b1q16_.begin(), b1q16_.end(), view);
      for (uint64_t bits = board; bits != 0; bits &= bits - 1)
        addRow(view, boardRow(__builtin_ctzll(bits)));
      for (int32_t i = 0; i < 6; ++i) {
        const int32_t c = count(rest, (3 * t + i) % 6);
        for (int32_t k = 0; k < c; ++k)
          addRow(view, reserveRow(i));
      }
    }
  }
  /// A child's two views from its parent's: the changed board bits, and the
  /// changed reserve count (a placement takes one piece)
  void updateAccumulators(const Acc *parent, uint64_t parent_board,
                          uint64_t parent_rest, uint64_t board, uint64_t rest,
                          Acc *child) const {
    for (int32_t t = 0; t < 2; ++t)
      updateView(parent + static_cast<size_t>(t) * h_, parent_board,
                 parent_rest, board, rest, t,
                 child + static_cast<size_t>(t) * h_);
  }
  /// Search score (scoreInt8) of a child from its parent's views, in one
  /// pass, building only the view of the child's side to move (lazy,
  /// entry 23)
  float scoreFromParent(const Acc *parent, uint64_t parent_board,
                        uint64_t parent_rest, uint64_t board,
                        uint64_t rest) const {
    const int32_t t = static_cast<int32_t>(rest & 1);
    // The rows to add and subtract (changed board bits, the changed reserve
    // count), then the activations straight from parent + rows, a block of
    // 64 at a time in registers: no accumulator stored (fused, entry 23)
    const Acc *add[72], *sub[72];  // at most 64 board bits + 4 reserves
    int32_t na = 0, ns = 0;
    for (uint64_t on = board & ~parent_board; on != 0; on &= on - 1)
      add[na++] = boardRow(__builtin_ctzll(on));
    for (uint64_t off = parent_board & ~board; off != 0; off &= off - 1)
      sub[ns++] = boardRow(__builtin_ctzll(off));
    for (int32_t piece = 0; piece < 6; ++piece) {
      int32_t d = count(rest, piece) - count(parent_rest, piece);
      const Acc *row = reserveRow((piece - 3 * t + 6) % 6);
      for (; d > 0; --d)
        add[na++] = row;
      for (; d < 0; ++d)
        sub[ns++] = row;
    }
    const Acc *base = parent + static_cast<size_t>(t) * h_;
    alignas(32) uint8_t act[kMaxHidden];
    for (int32_t j0 = 0; j0 < h_; j0 += 64) {
      Acc v[64];
      for (int32_t j = 0; j < 64; ++j)
        v[j] = base[j0 + j];
      for (int32_t k = 0; k < na; ++k)
        for (int32_t j = 0; j < 64; ++j)
          v[j] = static_cast<Acc>(v[j] + add[k][j0 + j]);
      for (int32_t k = 0; k < ns; ++k)
        for (int32_t j = 0; j < 64; ++j)
          v[j] = static_cast<Acc>(v[j] - sub[k][j0 + j]);
      for (int32_t j = 0; j < 64; ++j)
        act[j0 + j] = static_cast<uint8_t>(
            std::min(std::max(static_cast<int32_t>(v[j]) >> 3, 0), 127));
    }
    return kScoreScale * fromActivations(act);
  }

 private:
  static int32_t count(uint64_t rest, int32_t piece) {
    return static_cast<int32_t>((rest >> (4 + 4 * piece)) & 0xF);
  }
  const Acc *boardRow(int32_t bit) const {
    return board_rows_.data() + static_cast<size_t>(bit) * h_;
  }
  /// One piece in reserve slot i (slot 0-2: the side to move's)
  const Acc *reserveRow(int32_t slot) const {
    return reserve_rows_.data() + static_cast<size_t>(slot) * h_;
  }
  void updateView(const Acc *parent, uint64_t parent_board,
                  uint64_t parent_rest, uint64_t board, uint64_t rest,
                  int32_t t, Acc *__restrict out) const {
    std::copy(parent, parent + h_, out);
    for (uint64_t on = board & ~parent_board; on != 0; on &= on - 1)
      addRow(out, boardRow(__builtin_ctzll(on)));
    for (uint64_t off = parent_board & ~board; off != 0; off &= off - 1)
      subRow(out, boardRow(__builtin_ctzll(off)));
    for (int32_t piece = 0; piece < 6; ++piece) {
      const int32_t d = count(rest, piece) - count(parent_rest, piece);
      if (d == 0)
        continue;
      const Acc *row = reserveRow((piece - 3 * t + 6) % 6);
      for (int32_t k = 0; k < d; ++k)
        addRow(out, row);
      for (int32_t k = 0; k > d; --k)
        subRow(out, row);
    }
  }
  template <typename T>
  void addRow(T *__restrict acc, const T *__restrict row) const {
    for (int32_t j = 0; j < h_; ++j)
      acc[j] = static_cast<T>(acc[j] + row[j]);
  }
  template <typename T>
  void subRow(T *__restrict acc, const T *__restrict row) const {
    for (int32_t j = 0; j < h_; ++j)
      acc[j] = static_cast<T>(acc[j] - row[j]);
  }

  /// Layers 2 and 3 from layer 1's accumulator
  template <typename T> float fromAccumulator(const T *__restrict acc) const {
    alignas(32) uint8_t act_buf[kMaxHidden];
    uint8_t *__restrict act = act_buf;
    const int32_t h = h_;
    for (int32_t j = 0; j < h; ++j)
      act[j] = static_cast<uint8_t>(
          std::min(std::max(static_cast<int32_t>(acc[j]) >> 3, 0), 127));
    return fromActivations(act_buf);
  }

  /// Layers 2 and 3 from layer 1's clipped activations (uint8, 0-127)
  float fromActivations(const uint8_t *act_buf) const {
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

  /// Layer 3, before the tanh
  float output(const float *h2, float scale) const {
    float out = b3_;
    for (int32_t k = 0; k < 32; ++k)
      out += std::min(std::max(h2[k] * scale, 0.0F), 1.0F) * w3_[k];
    return out;
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
            static_cast<Acc>(4 * w1q_[static_cast<size_t>(b) * h_ + j]);
    reserve_rows_.resize(6 * static_cast<size_t>(h_));
    for (int32_t i = 0; i < 6; ++i)
      for (int32_t j = 0; j < h_; ++j)
        reserve_rows_[static_cast<size_t>(i) * h_ + j] =
            static_cast<Acc>(w1q_[static_cast<size_t>(64 + i) * h_ + j]);
    b1q16_.assign(b1q_.begin(), b1q_.end());
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
  std::vector<Acc> board_rows_, reserve_rows_, b1q16_;
  std::vector<int8_t> w2q_;
  float b2q_[32]{};
  float w2_scale_{1};
};

#endif
