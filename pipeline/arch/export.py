"""Export a supervised model (arch/sup.py, runs/sup/models/TAG.pt) for the driver.

    python arch/export.py runs/sup/models/s4_512x4.pt OUT.onnx

Compact interface (model.CompactIo): uint8 states in, fp32 value [N, 1] and
fp16 policy [N, 96] out. The policy is a softmax over all 96 moves; the
search keeps only legal moves and renormalizes, which for a model trained
with masking is exactly its masked distribution. Only models without extra
inputs (--lines, --legal-input) can be exported: the driver sends the 70
state inputs only.
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
import sup  # noqa: E402


class Probabilities(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        value, logits = self.net(x)
        return value.unsqueeze(1), torch.softmax(logits, 1)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    assert not a.get("lines") and not a.get("legal_input"), "extra inputs"
    if a["model"] == "mlp":
        net = sup.Mlp(70, a["width"], a["depth"])
    else:
        net = sup.ResMlp(
            70, a["width"], a["depth"], a["norm"], a.get("block", "pre")
        )
    net.load_state_dict(ck["model"])
    return net.eval()


def main():
    net = M.CompactIo(Probabilities(load(sys.argv[1]))).eval()
    torch.onnx.export(
        net,
        (torch.zeros(16, 70, dtype=torch.uint8),),
        sys.argv[2],
        input_names=["states"],
        output_names=["value", "policy"],
        dynamic_axes={k: {0: "n"} for k in ("states", "value", "policy")},
        opset_version=17,
        dynamo=False,
    )
    print(sys.argv[2])


if __name__ == "__main__":
    main()
