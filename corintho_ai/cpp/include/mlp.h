#ifndef MLP_H
#define MLP_H

#include <cstdint>

#include <string>
#include <vector>

/// @brief CPU forward pass of the policy/value network, in-process
/// @details The network is a plain MLP: hidden layers of fully connected +
/// ReLU, then a tanh value head and a softmax policy head, both reading the
/// last hidden layer. Batch normalization is expected to be folded into the
/// weights already (the tflite converter does this; bench/export_mlp.py writes
/// the file).
///
/// File format, all little-endian:
///   char[4]  magic "CMLP"
///   uint32   version (1)
///   uint32   input size, hidden width, number of hidden layers, moves
///   per hidden layer: float32 weights [out][in], float32 biases [out]
///   value head:       float32 weights [1][width], float32 bias [1]
///   policy head:      float32 weights [moves][width], float32 biases [moves]
///
/// Evaluation is thread-safe: the weights are read-only after loading.
class Mlp {
 public:
  explicit Mlp(const std::string &path);

  int32_t inputSize() const noexcept { return input_; }
  int32_t numMoves() const noexcept { return moves_; }

  /// @brief Evaluate num_rows states on the calling thread
  /// @param states num_rows x inputSize() floats
  /// @param evals Output, num_rows values in [-1, 1]
  /// @param probs Output, num_rows x numMoves() probabilities
  void evaluate(const float *states, int32_t num_rows, float *evals,
                float *probs) const;
  /// @brief Evaluate num_rows states, split across num_threads OpenMP threads
  /// @details num_threads 0 uses the OpenMP default.
  void evaluateParallel(const float *states, int32_t num_rows, float *evals,
                        float *probs, int32_t num_threads = 0) const;

 private:
  /// @brief Rows processed together, one layer at a time, so that a layer's
  /// weights stay in L1 across the block
  static constexpr int32_t kBlockRows = 16;

  /// @brief A fully connected layer, stored transposed as [in][out_pad]
  /// @details out_pad rounds out up to a multiple of 8 floats (one AVX2
  /// register) and the padding is zero, so the inner loop over outputs has no
  /// remainder.
  struct Layer {
    int32_t in{0};
    int32_t out{0};
    int32_t out_pad{0};
    std::vector<float> weights{};
    std::vector<float> biases{};
  };

  void evaluateBlock(const float *states, int32_t rows, float *evals,
                     float *probs) const;

  int32_t input_{0};
  int32_t width_{0};
  int32_t moves_{0};
  std::vector<Layer> hidden_{};
  Layer value_{};
  Layer policy_{};
};

#endif
