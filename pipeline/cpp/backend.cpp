#include "backend.h"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

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
  OrtBackend(const std::string &path, int32_t device_id, TensorRt trt)
      : path_{path}, trt_{trt} {
    // The environment (and its logger) must exist before any other ORT call
    Ort::Env &env = ortEnv();
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // The session runs on the GPU; one host thread is enough to drive it and
    // leaves the cores to the engine
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    const std::string device = std::to_string(device_id);
    if (trt != TensorRt::kOff) {
      // TensorRT fuses the whole network into a few kernels: 2.3-2.6x less
      // compute than the CUDA provider's one kernel per operation. Built
      // engines are cached next to the model, keyed by the model's contents.
      // Its "fp32" is not exact (it allows TF32); fp16 is further off.
      // Operations TensorRT cannot take fall back to the CUDA provider.
      const std::string cache = cacheDir(path);
      OrtTensorRTProviderOptionsV2 *opts = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&opts));
      const std::array<const char *, 8> keys{"device_id",
                                             "trt_fp16_enable",
                                             "trt_engine_cache_enable",
                                             "trt_engine_cache_path",
                                             "trt_timing_cache_enable",
                                             "trt_profile_min_shapes",
                                             "trt_profile_opt_shapes",
                                             "trt_profile_max_shapes"};
      const std::array<const char *, 8> vals{device.c_str(),
                                             trt == TensorRt::kFp16 ? "True"
                                                                    : "False",
                                             "True",
                                             cache.c_str(),
                                             "True",
                                             "states:1x70",
                                             "states:16000x70",
                                             "states:65536x70"};
      Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
          opts, keys.data(), vals.data(), keys.size()));
      options.AppendExecutionProvider_TensorRT_V2(*opts);
      Ort::GetApi().ReleaseTensorRTProviderOptions(opts);
    }
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda));
    const std::array<const char *, 2> keys{"device_id", "use_tf32"};
    const std::array<const char *, 2> vals{device.c_str(), "0"};
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
        cuda, keys.data(), vals.data(), keys.size()));
    options.AppendExecutionProvider_CUDA_V2(*cuda);
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda);
    session_ = std::make_unique<Ort::Session>(env, path.c_str(), options);
    pinned_ = std::make_unique<Ort::Allocator>(*session_, pinned_info_);
    // A compact model takes uint8 states (4x the engine's floats) and returns
    // the policy in fp16: 70 + 192 bytes a row cross the bus instead of 664.
    // The value stays fp32. Older models take and return fp32 throughout.
    compact_ = session_->GetInputTypeInfo(0)
                   .GetTensorTypeAndShapeInfo()
                   .GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }

  void evaluate(const float *states, int32_t rows, float *values,
                float *probs) override {
    // Stage through page-locked (pinned) host buffers: the GPU then moves the
    // data by DMA. With ordinary pageable buffers, CUDA copies through its own
    // staging buffer on the CPU, which competed with the engine's threads and
    // made each call ~2.6x slower while the engine was searching (entry 29).
    reserve(rows);
    const size_t in_n = static_cast<size_t>(rows) * kGameStateSize;
    const size_t policy_n = static_cast<size_t>(rows) * kNumMoves;
    const std::array<int64_t, 2> in_shape{rows, kGameStateSize};
    const std::array<int64_t, 2> value_shape{rows, 1};
    const std::array<int64_t, 2> policy_shape{rows, kNumMoves};
    Ort::IoBinding binding{*session_};
    if (compact_) {
      // Every input is a multiple of 0.25 in [0, 1], so 4x fits a byte
      // exactly; the model multiplies by 0.25 itself
      // (worklog/2026-09-25-nn-architectures, entry 01)
      auto *in = static_cast<uint8_t *>(pinned_in_);
      for (size_t i = 0; i < in_n; ++i)
        in[i] = static_cast<uint8_t>(states[i] * 4.0F + 0.5F);
      binding.BindInput("states",
                        Ort::Value::CreateTensor<uint8_t>(cpu_info_, in, in_n,
                                                          in_shape.data(), 2));
      binding.BindOutput("policy",
                         Ort::Value::CreateTensor<Ort::Float16_t>(
                             cpu_info_,
                             static_cast<Ort::Float16_t *>(pinned_policy_),
                             policy_n, policy_shape.data(), 2));
    } else {
      auto *in = static_cast<float *>(pinned_in_);
      std::copy(states, states + in_n, in);
      binding.BindInput("states",
                        Ort::Value::CreateTensor<float>(cpu_info_, in, in_n,
                                                        in_shape.data(), 2));
      binding.BindOutput("policy",
                         Ort::Value::CreateTensor<float>(
                             cpu_info_, static_cast<float *>(pinned_policy_),
                             policy_n, policy_shape.data(), 2));
    }
    binding.BindOutput(
        "value", Ort::Value::CreateTensor<float>(cpu_info_, pinned_value_,
                                                 static_cast<size_t>(rows),
                                                 value_shape.data(), 2));
    session_->Run(Ort::RunOptions{nullptr}, binding);
    std::copy(pinned_value_, pinned_value_ + rows, values);
    if (compact_) {
      // fp16 to fp32, eight at a time (F16C); 96 moves per row, so the total
      // is a multiple of eight
      const auto *half = static_cast<const uint16_t *>(pinned_policy_);
      for (size_t i = 0; i < policy_n; i += 8)
        _mm256_storeu_ps(probs + i,
                         _mm256_cvtph_ps(_mm_loadu_si128(
                             reinterpret_cast<const __m128i *>(half + i))));
    } else {
      const auto *p = static_cast<const float *>(pinned_policy_);
      std::copy(p, p + policy_n, probs);
    }
  }

  ~OrtBackend() override { release(); }

  std::string describe() const override {
    const char *kind = trt_ == TensorRt::kFp16   ? "ort-trt-fp16 "
                       : trt_ == TensorRt::kFp32 ? "ort-trt "
                                                 : "ort-cuda ";
    return kind + std::string{compact_ ? "compact " : ""} + path_;
  }

 private:
  void reserve(int32_t rows) {
    if (rows <= capacity_)
      return;
    release();
    capacity_ = std::max(rows, 2 * capacity_);
    // Sized for fp32 either way; a compact model uses the start of each
    auto get = [this](size_t floats) {
      return static_cast<float *>(pinned_->Alloc(floats * sizeof(float)));
    };
    pinned_in_ = get(static_cast<size_t>(capacity_) * kGameStateSize);
    pinned_value_ = get(static_cast<size_t>(capacity_));
    pinned_policy_ = get(static_cast<size_t>(capacity_) * kNumMoves);
  }
  void release() {
    for (void *p :
         {pinned_in_, static_cast<void *>(pinned_value_), pinned_policy_})
      if (p != nullptr)
        pinned_->Free(p);
    pinned_in_ = pinned_policy_ = nullptr;
    pinned_value_ = nullptr;
    capacity_ = 0;
  }

  static std::string cacheDir(const std::string &model) {
    const size_t slash = model.find_last_of('/');
    return (slash == std::string::npos ? std::string{"."}
                                       : model.substr(0, slash)) +
           "/trt_cache";
  }

  std::unique_ptr<Ort::Session> session_;
  std::string path_;
  TensorRt trt_;
  // Pinned buffers come from ORT's CudaPinned allocator but are bound as
  // plain CPU tensors: binding them as CudaPinned made ORT's CUDA provider
  // attempt a copy onto the same address. CUDA recognises page-locked memory
  // by address, so the host-device copies are still DMA.
  Ort::MemoryInfo pinned_info_{"CudaPinned", OrtDeviceAllocator, 0,
                               OrtMemTypeCPUOutput};
  Ort::MemoryInfo cpu_info_ =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::unique_ptr<Ort::Allocator> pinned_;
  void *pinned_in_{nullptr};
  float *pinned_value_{nullptr};
  void *pinned_policy_{nullptr};
  int32_t capacity_{0};
  bool compact_{false};
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
                                        int32_t device_id, TensorRt trt) {
  return std::make_unique<OrtBackend>(onnx_path, device_id, trt);
}

std::unique_ptr<Backend> makeBackend(const std::string &path,
                                     int32_t num_threads) {
  if (endsWith(path, ".mlp"))
    return makeCpuBackend(path, num_threads);
  for (const auto &[prefix, trt] : {std::pair{"trt16:", TensorRt::kFp16},
                                    std::pair{"trt:", TensorRt::kFp32}})
    if (path.rfind(prefix, 0) == 0 && endsWith(path, ".onnx"))
      return makeOrtBackend(path.substr(std::string{prefix}.size()), 0, trt);
  if (endsWith(path, ".onnx"))
    return makeOrtBackend(path, 0, TensorRt::kOff);
  throw std::runtime_error("unknown model type (want .mlp or .onnx): " + path);
}
