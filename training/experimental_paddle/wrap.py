#!/usr/bin/env python3
"""Adapt pinned Paddle probabilities to ocracle's grayscale/log-CTC contract.

This changes preprocessing/output representation, never trained parameters.
The original graph and dictionary remain subject to the upstream Apache-2.0
license; see LICENSE.paddle. The wrapper modifications are made by ocracle.
"""

from pathlib import Path
import hashlib
import json

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
MODEL_SHA256 = "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
CONFIG_SHA256 = "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1"


def main():
    """Write an ONNX adapter and an experimental configuration template."""
    source = ROOT / "cache/inference.onnx"
    config_source = ROOT / "cache/inference.yml"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == MODEL_SHA256
    assert hashlib.sha256(config_source.read_bytes()).hexdigest() == CONFIG_SHA256
    model = onnx.load(source)
    dictionary = yaml.safe_load(config_source.read_text())["PostProcess"]["character_dict"]
    assert len(dictionary) == 18708 and len(set(dictionary)) == 18708
    assert all(len(character) == 1 for character in dictionary)
    assert " " not in dictionary
    # Only the input edge changes. All original weights/nodes are retained.
    old_input = model.graph.input[0].name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old_input:
                node.input[index] = "adapter_padded"
    constants = {
        "adapter_two": np.array(2, dtype=np.float32),
        "adapter_repeats": np.array([1, 3, 1, 1], dtype=np.int64),
        "adapter_axis3": np.array(3, dtype=np.int64),
        "adapter_minwidth": np.array(320, dtype=np.int64),
        "adapter_zero_i64": np.array(0, dtype=np.int64),
        "adapter_seven_zeros": np.zeros(7, dtype=np.int64),
        "adapter_zero_f32": np.array(0, dtype=np.float32),
        "adapter_log_min": np.array(1e-30, dtype=np.float32),
        "adapter_log_max": np.array(1, dtype=np.float32),
    }
    for name, value in constants.items():
        model.graph.initializer.append(numpy_helper.from_array(value, name))
    n = helper.make_node
    prefix = [
        n("Mul", ["input", "adapter_two"], ["adapter_normalized"]),
        n("Tile", ["adapter_normalized", "adapter_repeats"], ["adapter_bgr"]),
        n("Shape", ["input"], ["adapter_shape"]),
        n("Gather", ["adapter_shape", "adapter_axis3"], ["adapter_width"], axis=0),
        n("Sub", ["adapter_minwidth", "adapter_width"], ["adapter_delta"]),
        n("Less", ["adapter_width", "adapter_minwidth"], ["adapter_narrow"]),
        n("Where", ["adapter_narrow", "adapter_delta", "adapter_zero_i64"], ["adapter_pad_w"]),
        n("Unsqueeze", ["adapter_pad_w"], ["adapter_pad_w_1d"], axes=[0]),
        n("Concat", ["adapter_seven_zeros", "adapter_pad_w_1d"], ["adapter_pads"], axis=0),
        n("Pad", ["adapter_bgr", "adapter_pads", "adapter_zero_f32"], ["adapter_padded"], mode="constant"),
    ]
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(prefix + original_nodes)
    old_output = model.graph.output[0].name
    model.graph.node.extend([
        n("Clip", [old_output, "adapter_log_min", "adapter_log_max"], ["adapter_probabilities"]),
        n("Log", ["adapter_probabilities"], ["adapter_log_ntc"]),
        n("Transpose", ["adapter_log_ntc"], ["log_probs"], perm=[1, 0, 2]),
    ])
    del model.graph.input[:]
    model.graph.input.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, 48, "W"]))
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, ["T", "N", 18710]))
    model.doc_string = "ocracle experimental grayscale/log-CTC adapter; original Paddle weights unchanged."
    onnx.checker.check_model(model)
    destination = ROOT / "cache/paddle-v6-small-gray-logctc.onnx"
    onnx.save(model, destination)
    config = {
        "alphabet": "".join(dictionary) + " ",
        "input_height": 48,
        "model_name": "Paddle PP-OCRv6 small (experimental grayscale adapter)",
        "architecture": "LCNetV4 + LightSVTR + CTC",
        "language": "mul",
    }
    (ROOT / "model-config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "classes": 18710}))


if __name__ == "__main__":
    main()
