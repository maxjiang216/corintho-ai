#include "backend.h"

#include <array>
#include <stdexcept>

#include "mlp.h"
#include "onnxruntime_cxx_api.h"
#include "util.h"

namespace {

class CpuBackend : public Backend {
 public:
  CpuBackend(const std::string &path, int32_t num_threads)
      : mlp_{path}, path_{path}, num_threads_{num_threads} {
    if (mlp_.inputSize() != kGameStateSize || mlp_.numMoves() != kNumMoves)
      throw std::runtime_error("model shape does not match the engine: " +
                               path);
  }
  void evaluate(const float *states, int32_t rows, float *values,
                float *probs) override {
    mlp_.evaluateParallel(states, rows, values, probs, num_threads_);
  }
  std::string describe() const override { return "cpu-mlp " + path_; }

 private:
  Mlp mlp_;
  std::string path_;
  int32_t num_threads_;
};

Ort::Env &ortEnv() {
  // Deliberately never destroyed. A function-local static Env is destroyed
  // during exit, after the CUDA provider's own teardown, and that corrupted
  // the heap at exit in every optimized build (worklog entry 29). The OS
  // reclaims everything at process exit anyway.
  static Ort::Env *env = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "corintho"};
  return *env;
}

class OrtBackend : public Backend {
 public:
  OrtBackend(const std::string &path, int32_t device_id) : path_{path} {
    // The environment (and its logger) must exist before any other ORT call
    Ort::Env &env = ortEnv();
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // The session runs on the GPU; one host thread is enough to drive it and
    // leaves the cores to the engine
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
    const std::string device = std::to_string(device_id);
    const std::array<const char *, 2> keys{"device_id", "use_tf32"};
    const std::array<const char *, 2> vals{device.c_str(), "0"};
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
        cuda, keys.data(), vals.data(), keys.size()));
    options.AppendExecutionProvider_CUDA_V2(*cuda);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
    session_ = std::make_unique<Ort::Session>(env, path.c_str(), options);
  }

  void evaluate(const float *states, int32_t rows, float *values,
                float *probs) override {
    const Ort::MemoryInfo cpu =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> in_shape{rows, kGameStateSize};
    const std::array<int64_t, 2> value_shape{rows, 1};
    const std::array<int64_t, 2> policy_shape{rows, kNumMoves};
    // ORT does not write through the input, but its API takes non-const
    Ort::Value input = Ort::Value::CreateTensor<float>(
        cpu, const_cast<float *>(states),
        static_cast<size_t>(rows) * kGameStateSize, in_shape.data(), 2);
    std::array<Ort::Value, 2> outputs{
        Ort::Value::CreateTensor<float>(cpu, values, static_cast<size_t>(rows),
                                        value_shape.data(), 2),
        Ort::Value::CreateTensor<float>(
            cpu, probs, static_cast<size_t>(rows) * kNumMoves,
            policy_shape.data(), 2)};
    const std::array<const char *, 1> in_names{"states"};
    const std::array<const char *, 2> out_names{"value", "policy"};
    session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &input, 1,
                  out_names.data(), outputs.data(), 2);
  }
  std::string describe() const override { return "ort-cuda " + path_; }

 private:
  std::unique_ptr<Ort::Session> session_;
  std::string path_;
};

bool endsWith(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::unique_ptr<Backend> makeCpuBackend(const std::string &mlp_path,
                                        int32_t num_threads) {
  return std::make_unique<CpuBackend>(mlp_path, num_threads);
}

std::unique_ptr<Backend> makeOrtBackend(const std::string &onnx_path,
                                        int32_t device_id) {
  return std::make_unique<OrtBackend>(onnx_path, device_id);
}

std::unique_ptr<Backend> makeBackend(const std::string &path,
                                     int32_t num_threads) {
  if (endsWith(path, ".mlp"))
    return makeCpuBackend(path, num_threads);
  if (endsWith(path, ".onnx"))
    return makeOrtBackend(path, 0);
  throw std::runtime_error("unknown model type (want .mlp or .onnx): " + path);
}
