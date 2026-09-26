# Does model_93.tflite compute the same function as the Keras SavedModel of gen_93?
import sys

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

saved, tfl = sys.argv[1], sys.argv[2]
m = tf.saved_model.load(saved)
f = m.signatures["serving_default"]
print(
    "signature outputs:", {k: v.shape for k, v in f.structured_outputs.items()}
)
rng = np.random.default_rng(3)
x = rng.integers(0, 2, (4096, 70)).astype(np.float32)
out = f(tf.constant(x))
keys = sorted(out.keys())
ko = {k: out[k].numpy() for k in keys}
it = Interpreter(model_path=tfl)
inp = it.get_input_details()[0]["index"]
it.resize_tensor_input(inp, [len(x), 70])
it.allocate_tensors()
it.set_tensor(inp, x)
it.invoke()
to = {
    d["shape"][-1]: it.get_tensor(d["index"]) for d in it.get_output_details()
}
for k in keys:
    a = ko[k]
    b = to[a.shape[-1]]
    print(k, a.shape, "max |keras - tflite|", float(np.abs(a - b).max()))
