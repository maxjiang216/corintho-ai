"""Evaluate game states with a real tflite model for selfplay_nn.

Spawned by selfplay_nn with its stdin/stdout connected to pipes. Protocol, all
little-endian, repeated until EOF:

  request:  int32 n, then n * 70 float32 game states
  response: n float32 evaluations, then n * 96 float32 move probabilities

The model's outputs are used exactly as main.pyx uses Keras predict():
res[0] (value) flattened into evaluations and res[1] (softmax) into probs.

Needs a LiteRT runtime (pip install ai-edge-litert), which the training
environment does not use; it is only for engine measurements.
"""

import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

STATE = 70
MOVES = 96


def read_exact(stream, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def main():
    interpreter = Interpreter(model_path=sys.argv[1])
    inp = interpreter.get_input_details()[0]["index"]
    outs = {d["shape"][-1]: d["index"] for d in interpreter.get_output_details()}
    value_idx, prob_idx = outs[1], outs[MOVES]
    size = 0
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        header = read_exact(stdin, 4)
        if header is None:
            return
        n = int(np.frombuffer(header, dtype="<i4")[0])
        states = np.frombuffer(read_exact(stdin, n * STATE * 4), dtype="<f4")
        if n != size:
            interpreter.resize_tensor_input(inp, [n, STATE])
            interpreter.allocate_tensors()
            size = n
        interpreter.set_tensor(inp, states.reshape(n, STATE))
        interpreter.invoke()
        stdout.write(interpreter.get_tensor(value_idx).astype("<f4").tobytes())
        stdout.write(interpreter.get_tensor(prob_idx).astype("<f4").tobytes())
        stdout.flush()


if __name__ == "__main__":
    main()
