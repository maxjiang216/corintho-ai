#!/usr/bin/env bash
# Sets up the GPU pipeline's local environment (worklog entry 29):
#   .venv/         PyTorch, ONNX Runtime (Python), numpy, onnx; the pip wheels
#                  also bring the CUDA 13 and cuDNN 9 libraries, so no system
#                  CUDA toolkit is needed, only the NVIDIA driver
#                  and TensorRT 10 (4.3 GB), for trt: models (entry 30)
#   third_party/   the ONNX Runtime C++ release, CUDA 13 build, for the driver
# Then builds build/corintho_play. Needs uv, clang-20 and libomp-20-dev.
set -euo pipefail
cd "$(dirname "$0")"

ORT_VERSION=1.30.0
ORT_NAME=onnxruntime-linux-x64-gpu_cuda13-$ORT_VERSION
ORT_SHA256=382d79133112388cf94ce5855789b7c9bef12bef76a08b6b277e5a317213adcd

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
# Versions verified together on 2026-09-25 (torch 2.14.0+cu130)
VIRTUAL_ENV=.venv uv pip install torch numpy onnx onnxscript "onnxruntime-gpu==$ORT_VERSION" \
  "tensorrt-cu13==10.16.1.11"  # ORT 1.30's TensorRT provider needs TensorRT 10

if [ ! -d "third_party/$ORT_NAME" ]; then
  mkdir -p third_party
  curl -sSL -o third_party/ort.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_NAME.tgz"
  echo "$ORT_SHA256  third_party/ort.tgz" | sha256sum -c -
  tar xzf third_party/ort.tgz -C third_party
  rm third_party/ort.tgz
fi

make -s
echo "built build/corintho_play"

# The starting model: gen_93, the last cloud generation, converted once from
# its Keras SavedModel (keras_to_npz.py, needs TensorFlow) into models/gen_93.npz
if [ ! -f models/gen_93.onnx ]; then
  .venv/bin/python model.py convert models/gen_93.npz models/gen_93
fi
