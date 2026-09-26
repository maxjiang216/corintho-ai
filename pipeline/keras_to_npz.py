"""One-time conversion of a Keras SavedModel generation to a PyTorch-loadable .npz.

    python keras_to_npz.py ../generations/gen_93/model gen_93.npz

Needs TensorFlow and tf_keras (a throwaway venv is enough; see pipeline/README.md).
The pipeline itself never imports TensorFlow.

Unlike the tflite file, the SavedModel keeps each BatchNormalization separate,
with its moving statistics, which is what training needs: model.py rebuilds
the same Dense -> ReLU -> BatchNorm stack and loads these arrays into it.
Adam's moment estimates are not carried over; the optimizer starts fresh.

Arrays written, in layer order:
  dense{i}_w [in, out], dense{i}_b [out]                     i = 0..11
  bn{i}_gamma, bn{i}_beta, bn{i}_mean, bn{i}_var [100]       i = 0..11
  value_w [100, 1], value_b [1], policy_w [100, 96], policy_b [96]
  bn_epsilon, learning_rate (scalars)
"""
import sys

import numpy as np
import tf_keras as keras


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = keras.models.load_model(src)
    out = {}
    dense, bns, heads = [], [], []
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Dense":
            (heads if layer.units in (1, 96) else dense).append(layer)
        elif kind == "BatchNormalization":
            bns.append(layer)
        elif kind not in ("InputLayer", "Activation"):
            sys.exit(f"unexpected layer {layer.name} ({kind})")
    assert len(dense) == len(bns) == 12, (len(dense), len(bns))
    for i, (d, bn) in enumerate(zip(dense, bns)):
        assert d.activation.__name__ == "linear"
        w, b = d.get_weights()
        out[f"dense{i}_w"], out[f"dense{i}_b"] = w, b
        gamma, beta, mean, var = bn.get_weights()
        out[f"bn{i}_gamma"], out[f"bn{i}_beta"] = gamma, beta
        out[f"bn{i}_mean"], out[f"bn{i}_var"] = mean, var
        eps = bn.epsilon
    for d in heads:
        w, b = d.get_weights()
        name = "value" if d.units == 1 else "policy"
        assert d.activation.__name__ == (
            "tanh" if name == "value" else "softmax"
        )
        out[f"{name}_w"], out[f"{name}_b"] = w, b
    out["bn_epsilon"] = np.float32(eps)
    out["learning_rate"] = np.float32(
        keras.backend.get_value(model.optimizer.learning_rate)
    )

    # Reference outputs for model.py to check its reconstruction against
    rng = np.random.default_rng(0)
    x = (rng.random((256, 70)) < 0.3).astype(np.float32)
    value, policy = model(x, training=False)
    out["check_x"], out["check_value"], out["check_policy"] = (
        x,
        np.asarray(value),
        np.asarray(policy),
    )
    np.savez(dst, **out)
    print(f"{dst}: 12 x 100 hidden, bn eps {eps}, lr {out['learning_rate']}")


if __name__ == "__main__":
    main()
