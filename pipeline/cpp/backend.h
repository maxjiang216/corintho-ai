#ifndef PIPELINE_BACKEND_H
#define PIPELINE_BACKEND_H

#include <cstdint>

#include <memory>
#include <string>

/// @brief A policy/value network that evaluates batches of game states
/// @details Rows are kGameStateSize floats in; one value in [-1, 1] and
/// kNumMoves probabilities out, in the same row order. Implementations may
/// keep state between calls (device buffers), so one Backend serves one
/// caller thread.
class Backend {
 public:
  virtual ~Backend() = default;
  virtual void evaluate(const float *states, int32_t rows, float *values,
                        float *probs) = 0;
  virtual std::string describe() const = 0;
};

/// @brief The in-process CPU network (corintho_ai/cpp/include/mlp.h), from a
/// .mlp file, split across num_threads OpenMP threads
std::unique_ptr<Backend> makeCpuBackend(const std::string &mlp_path,
                                        int32_t num_threads);

/// @brief ONNX Runtime with the CUDA execution provider, from a .onnx file
/// @details The model has one input, states [N, 70] float32, and two outputs,
/// value [N, 1] and policy [N, 96] float32 (pipeline/model.py exports it).
/// fp32 throughout: TF32 and fp16 put probabilities off by up to 0.18 with
/// this network (worklog entry 20), so TF32 is disabled explicitly.
enum class TensorRt { kOff, kFp32, kFp16 };

std::unique_ptr<Backend> makeOrtBackend(const std::string &onnx_path,
                                        int32_t device_id, TensorRt trt);

/// @brief Pick by file extension: .mlp -> CPU, .onnx -> ONNX Runtime CUDA
std::unique_ptr<Backend> makeBackend(const std::string &path,
                                     int32_t num_threads);

#endif
