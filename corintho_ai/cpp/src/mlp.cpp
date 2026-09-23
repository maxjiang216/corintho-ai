#include "mlp.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t kPad = 8;

int32_t padded(int32_t n) {
  return (n + kPad - 1) / kPad * kPad;
}

void readExact(std::ifstream &file, void *data, size_t size,
               const std::string &path) {
  file.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  if (!file)
    throw std::runtime_error("Mlp: truncated file " + path);
}

#if defined(__AVX2__) && defined(__FMA__)

/// @brief One register tile: kRows rows x kVecs x 8 outputs, starting at
/// output column col
/// @details The kRows * kVecs accumulators stay in registers for the whole
/// loop over inputs. Each step loads kVecs weight vectors and broadcasts kRows
/// inputs, then does kRows * kVecs FMAs. With 4 x 3 that is 12 accumulators
/// and 3 weight registers out of 16, and 7 loads per 12 FMAs, so the loop is
/// bound by FMA throughput rather than loads.
template <int32_t kRows, int32_t kVecs>
void denseTile(const float *in, int32_t in_stride, int32_t in_size,
               const float *weights, const float *biases, int32_t out_pad,
               float *out, int32_t col, bool relu) {
  __m256 acc[kRows][kVecs];
  for (int32_t v = 0; v < kVecs; ++v) {
    const __m256 b = _mm256_loadu_ps(biases + col + 8 * v);
    for (int32_t r = 0; r < kRows; ++r)
      acc[r][v] = b;
  }
  for (int32_t i = 0; i < in_size; ++i) {
    const float *w = weights + static_cast<size_t>(i) * out_pad + col;
    __m256 wv[kVecs];
    for (int32_t v = 0; v < kVecs; ++v)
      wv[v] = _mm256_loadu_ps(w + 8 * v);
    for (int32_t r = 0; r < kRows; ++r) {
      const __m256 x =
          _mm256_broadcast_ss(in + static_cast<size_t>(r) * in_stride + i);
      for (int32_t v = 0; v < kVecs; ++v)
        acc[r][v] = _mm256_fmadd_ps(x, wv[v], acc[r][v]);
    }
  }
  const __m256 zero = _mm256_setzero_ps();
  for (int32_t r = 0; r < kRows; ++r)
    for (int32_t v = 0; v < kVecs; ++v)
      _mm256_storeu_ps(out + static_cast<size_t>(r) * out_pad + col + 8 * v,
                       relu ? _mm256_max_ps(acc[r][v], zero) : acc[r][v]);
}

/// @brief All output columns for kRows rows: tiles of 3 vectors, then single
/// vectors for the remainder (100 outputs pad to 13 vectors = 4 x 3 + 1)
template <int32_t kRows>
void denseRows(const float *in, int32_t in_stride, int32_t in_size,
               const float *weights, const float *biases, int32_t out_pad,
               float *out, bool relu) {
  int32_t col = 0;
  for (; col + 24 <= out_pad; col += 24)
    denseTile<kRows, 3>(in, in_stride, in_size, weights, biases, out_pad, out,
                        col, relu);
  for (; col < out_pad; col += 8)
    denseTile<kRows, 1>(in, in_stride, in_size, weights, biases, out_pad, out,
                        col, relu);
}

/// @brief out[r][o] = bias[o] + sum_i in[r][i] * weights[i][o], for a block
/// @details in has row stride in_stride, out has row stride out_pad (a
/// multiple of 8). Rows go four at a time, then one at a time.
void dense(const float *in, int32_t in_stride, int32_t rows, int32_t in_size,
           const float *weights, const float *biases, int32_t out_pad,
           float *out, bool relu) {
  int32_t r = 0;
  for (; r + 4 <= rows; r += 4)
    denseRows<4>(in + static_cast<size_t>(r) * in_stride, in_stride, in_size,
                 weights, biases, out_pad,
                 out + static_cast<size_t>(r) * out_pad, relu);
  for (; r < rows; ++r)
    denseRows<1>(in + static_cast<size_t>(r) * in_stride, in_stride, in_size,
                 weights, biases, out_pad,
                 out + static_cast<size_t>(r) * out_pad, relu);
}

#else

/// @brief out[r][o] = bias[o] + sum_i in[r][i] * weights[i][o], for a block
/// @details Portable fallback. in has row stride in_stride, out has row stride
/// out_pad. The loop over outputs is the inner one, so it vectorizes with no
/// reduction.
void dense(const float *in, int32_t in_stride, int32_t rows, int32_t in_size,
           const float *weights, const float *biases, int32_t out_pad,
           float *out, bool relu) {
  for (int32_t r = 0; r < rows; ++r) {
    const float *x = in + static_cast<size_t>(r) * in_stride;
    float *y = out + static_cast<size_t>(r) * out_pad;
    std::memcpy(y, biases, sizeof(float) * out_pad);
    for (int32_t i = 0; i < in_size; ++i) {
      const float xi = x[i];
      const float *w = weights + static_cast<size_t>(i) * out_pad;
#pragma omp simd
      for (int32_t o = 0; o < out_pad; ++o)
        y[o] += xi * w[o];
    }
    if (relu) {
#pragma omp simd
      for (int32_t o = 0; o < out_pad; ++o)
        y[o] = std::max(y[o], 0.0F);
    }
  }
}

#endif

}  // namespace

Mlp::Mlp(const std::string &path) {
  std::ifstream file{path, std::ios::binary};
  if (!file)
    throw std::runtime_error("Mlp: cannot open " + path);
  char magic[4];
  readExact(file, magic, sizeof(magic), path);
  if (std::memcmp(magic, "CMLP", 4) != 0)
    throw std::runtime_error("Mlp: not a CMLP file: " + path);
  uint32_t header[5];
  readExact(file, header, sizeof(header), path);
  if (header[0] != 1)
    throw std::runtime_error("Mlp: unsupported version in " + path);
  input_ = static_cast<int32_t>(header[1]);
  width_ = static_cast<int32_t>(header[2]);
  const auto depth = static_cast<int32_t>(header[3]);
  moves_ = static_cast<int32_t>(header[4]);
  if (input_ <= 0 || width_ <= 0 || depth <= 0 || moves_ <= 0)
    throw std::runtime_error("Mlp: bad dimensions in " + path);

  auto read_layer = [&](int32_t in, int32_t out) {
    Layer layer;
    layer.in = in;
    layer.out = out;
    layer.out_pad = padded(out);
    std::vector<float> raw(static_cast<size_t>(out) * in);
    readExact(file, raw.data(), sizeof(float) * raw.size(), path);
    layer.weights.assign(static_cast<size_t>(in) * layer.out_pad, 0.0F);
    for (int32_t o = 0; o < out; ++o)
      for (int32_t i = 0; i < in; ++i)
        layer.weights[static_cast<size_t>(i) * layer.out_pad + o] =
            raw[static_cast<size_t>(o) * in + i];
    layer.biases.assign(layer.out_pad, 0.0F);
    readExact(file, layer.biases.data(), sizeof(float) * out, path);
    return layer;
  };
  for (int32_t l = 0; l < depth; ++l)
    hidden_.push_back(read_layer(l == 0 ? input_ : width_, width_));
  value_ = read_layer(width_, 1);
  policy_ = read_layer(width_, moves_);
  if (file.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("Mlp: trailing data in " + path);
}

void Mlp::evaluate(const float *states, int32_t num_rows, float *evals,
                   float *probs) const {
  for (int32_t start = 0; start < num_rows; start += kBlockRows) {
    const int32_t rows = std::min(kBlockRows, num_rows - start);
    evaluateBlock(states + static_cast<size_t>(start) * input_, rows,
                  evals + start, probs + static_cast<size_t>(start) * moves_);
  }
}

void Mlp::evaluateParallel(const float *states, int32_t num_rows, float *evals,
                           float *probs, int32_t num_threads) const {
  const int32_t num_blocks = (num_rows + kBlockRows - 1) / kBlockRows;
  if (num_threads <= 0)
    num_threads = omp_get_max_threads();
#pragma omp parallel for schedule(dynamic, 4) num_threads(num_threads)
  for (int32_t b = 0; b < num_blocks; ++b) {
    const int32_t start = b * kBlockRows;
    const int32_t rows = std::min(kBlockRows, num_rows - start);
    evaluateBlock(states + static_cast<size_t>(start) * input_, rows,
                  evals + start, probs + static_cast<size_t>(start) * moves_);
  }
}

void Mlp::evaluateBlock(const float *states, int32_t rows, float *evals,
                        float *probs) const {
  // Two ping-pong activation buffers, wide enough for any layer's output
  const int32_t stride = std::max(padded(width_), policy_.out_pad);
  thread_local std::vector<float> buf_a, buf_b;
  const size_t size = static_cast<size_t>(kBlockRows) * stride;
  if (buf_a.size() < size) {
    buf_a.assign(size, 0.0F);
    buf_b.assign(size, 0.0F);
  }
  const float *in = states;
  int32_t in_stride = input_;
  float *out = buf_a.data();
  for (const Layer &layer : hidden_) {
    dense(in, in_stride, rows, layer.in, layer.weights.data(),
          layer.biases.data(), layer.out_pad, out, true);
    in = out;
    in_stride = layer.out_pad;
    out = (out == buf_a.data()) ? buf_b.data() : buf_a.data();
  }

  dense(in, in_stride, rows, width_, value_.weights.data(),
        value_.biases.data(), value_.out_pad, out, false);
  for (int32_t r = 0; r < rows; ++r)
    evals[r] = std::tanh(out[static_cast<size_t>(r) * value_.out_pad]);

  dense(in, in_stride, rows, width_, policy_.weights.data(),
        policy_.biases.data(), policy_.out_pad, out, false);
  for (int32_t r = 0; r < rows; ++r) {
    const float *logits = out + static_cast<size_t>(r) * policy_.out_pad;
    float *p = probs + static_cast<size_t>(r) * moves_;
    const float max_logit = *std::max_element(logits, logits + moves_);
    float sum = 0.0F;
    for (int32_t m = 0; m < moves_; ++m) {
      p[m] = std::exp(logits[m] - max_logit);
      sum += p[m];
    }
    const float scale = 1.0F / sum;
    for (int32_t m = 0; m < moves_; ++m)
      p[m] *= scale;
  }
}
