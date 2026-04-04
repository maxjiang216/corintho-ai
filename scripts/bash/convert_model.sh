#!/usr/bin/env bash
# Convert the trained TFLite model to ONNX for the browser (ONNX Runtime Web).
# Output: web/model.onnx — commit alongside the other web/ assets.
#
# Usage:
#   pip install tf2onnx
#   bash scripts/bash/convert_model.sh

set -euo pipefail

TFLITE_SRC="corintho_ai/python/tflite_model.tflite"
ONNX_OUT="web/model.onnx"

python -m tf2onnx.convert \
  --tflite "$TFLITE_SRC" \
  --output "$ONNX_OUT" \
  --opset 13

echo "Wrote $ONNX_OUT"
