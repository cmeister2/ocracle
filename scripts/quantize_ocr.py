#!/usr/bin/env python3
"""Export an experimental dynamic-int8 OCR model without changing registered models.

Requires the onnx and onnxruntime versions in training/requirements.txt. The
output is an experiment: successful conversion establishes neither runtime
compatibility, OCR accuracy nor a performance improvement.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile


def digest(path):
    """Return the identity of the complete artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contract(model):
    """Capture public tensor types and shapes without inferring new shapes."""
    def describe(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "element_type": tensor.elem_type,
            "shape": [
                dimension.dim_param
                if dimension.HasField("dim_param")
                else dimension.dim_value
                if dimension.HasField("dim_value")
                else None
                for dimension in tensor.shape.dim
            ],
        }

    return {
        "inputs": [describe(value) for value in model.graph.input],
        "outputs": [describe(value) for value in model.graph.output],
    }


def describe(path, model):
    """Describe exported bytes and operators for later benchmark comparison."""
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "opsets": {entry.domain: entry.version for entry in model.opset_import},
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        **contract(model),
    }


def main():
    """Check artifact boundaries, export to a temporary file, then publish it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-sha256", required=True, help="Expected input artifact identity")
    parser.add_argument("--report", required=True, type=Path, help="New JSON experiment report")
    parser.add_argument("--matmul-only", action="store_true", help="Leave convolutions in float32")
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Use 7-bit weights to avoid saturation on x86 CPUs without VNNI",
    )
    parser.add_argument(
        "--uint8-weights",
        action="store_true",
        help="Alternative experiment; RTen's preferred weights are signed int8",
    )
    args = parser.parse_args()
    source, output, report = (path.resolve() for path in (args.source, args.output, args.report))
    if len({source, output, report}) != 3:
        parser.error("source, output and report must be different paths")
    for path in (output, report):
        if path.exists():
            parser.error(f"refusing to overwrite existing artifact: {path}")
        if not path.parent.is_dir():
            parser.error(f"output directory does not exist: {path.parent}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        parser.error("--source-sha256 must be a lowercase SHA-256 hex digest")
    if digest(source) != args.source_sha256:
        parser.error("source SHA-256 does not match")

    import onnx
    import onnxruntime
    from onnxruntime.quantization import QuantType, quantize_dynamic

    original = onnx.load(source, load_external_data=False)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in original.graph.initializer):
        parser.error("external tensor files require independent identity verification")
    ops = ["MatMul"] if args.matmul_only else ["Conv", "MatMul"]
    weight_type = QuantType.QUInt8 if args.uint8_weights else QuantType.QInt8
    with tempfile.TemporaryDirectory(prefix="ocr-int8-", dir=output.parent) as directory:
        temporary = Path(directory) / "quantized.onnx"
        quantize_dynamic(
            source,
            temporary,
            op_types_to_quantize=ops,
            per_channel=True,
            reduce_range=args.reduce_range,
            weight_type=weight_type,
            extra_options={"MatMulConstBOnly": True},
        )
        quantized = onnx.load(temporary)
        onnx.checker.check_model(quantized)
        if contract(original) != contract(quantized):
            raise ValueError("quantization changed the public tensor contract")
        operators = Counter(node.op_type for node in quantized.graph.node)
        if not (operators["ConvInteger"] or operators["MatMulInteger"]):
            raise ValueError("no integer compute operators were produced")
        if digest(source) != args.source_sha256:
            raise ValueError("source changed during conversion")
        evidence = {
            "schema_version": 1,
            "purpose": "experimental export; runtime, accuracy and latency require separate validation",
            "tool_versions": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
            "settings": {
                "mode": "dynamic",
                "operators": ops,
                "weight_type": str(weight_type),
                "per_channel_requested": True,
                "reduce_range": args.reduce_range,
                "MatMulConstBOnly": True,
            },
            "source": describe(source, original),
            "output": {**describe(temporary, quantized), "file": output.name},
        }
        # Exclusive creation also prevents overwrites if another process created
        # the requested destination while quantization was running.
        with output.open("xb") as stream:
            stream.write(temporary.read_bytes())
        with report.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
