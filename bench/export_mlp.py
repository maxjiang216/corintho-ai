"""Export a trained tflite model to the flat .mlp format read by Mlp (mlp.h).

  python export_mlp.py model_93.tflite model_93.mlp

The tflite converter has already folded each BatchNormalization into the Dense
layer after it, so the network at inference time is a plain MLP:

  12 x (fully connected + ReLU), 70 -> 100 -> ... -> 100
  value head:  fully connected to 1, then tanh
  policy head: fully connected to 96, then softmax

File format, all little-endian (see mlp.h, which is the authority):

  char[4]  magic "CMLP"
  uint32   version (1)
  uint32   input size, hidden width, number of hidden layers, number of moves
  per hidden layer: float32 weights [out][in], float32 biases [out]
  value head:       float32 weights [1][width], float32 bias [1]
  policy head:      float32 weights [moves][width], float32 biases [moves]

Needs a LiteRT runtime (pip install ai-edge-litert).
"""

import struct
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def main():
    src, dst = sys.argv[1], sys.argv[2]
    interpreter = Interpreter(model_path=src)
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()

    def tensor(index):
        return interpreter.get_tensor(index)

    layers = []  # (weights [out][in], bias [out]) in execution order
    heads = {}
    for op in ops:
        if op["op_name"] != "FULLY_CONNECTED":
            continue
        _, w_idx, b_idx = op["inputs"]
        w, b = tensor(w_idx), tensor(b_idx)
        if w.shape[0] == 1:
            heads["value"] = (w, b)
        elif w.shape[0] == 96:
            heads["policy"] = (w, b)
        else:
            layers.append((w, b))

    # Check the structure is the one described above before trusting it
    names = [op["op_name"] for op in ops if op["op_name"] != "DELEGATE"]
    n = len(layers)
    expected = ["FULLY_CONNECTED"] * (n + 1) + ["TANH", "FULLY_CONNECTED", "SOFTMAX"]
    if names != expected:
        sys.exit(f"unexpected op sequence: {names}")
    input_size = layers[0][0].shape[1]
    width = layers[0][0].shape[0]
    moves = heads["policy"][0].shape[0]
    for w, b in layers[1:]:
        assert w.shape == (width, width) and b.shape == (width,)
    assert heads["value"][0].shape == (1, width)
    assert heads["policy"][0].shape == (moves, width)

    with open(dst, "wb") as f:
        f.write(b"CMLP")
        f.write(struct.pack("<5I", 1, input_size, width, n, moves))
        for w, b in layers + [heads["value"], heads["policy"]]:
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())
    print(f"{dst}: input {input_size}, {n} x {width} hidden, {moves} moves")


if __name__ == "__main__":
    main()
