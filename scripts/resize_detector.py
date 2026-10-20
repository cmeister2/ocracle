#!/usr/bin/env python3
"""Specialize the pinned fully convolutional OCR detector for a fixed input size.

Only input/output metadata and twelve U-Net crop endpoints change. This is an
experimental speed/accuracy tradeoff, not a newly trained model. Requires onnx.
"""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper

SOURCE_SHA256 = 'a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe'


def specialize(source: Path, output: Path, height: int, width: int) -> dict:
    """Validate the known graph then rewrite its spatial shape and crop limits."""
    if not (64 <= height <= 1600 and 64 <= width <= 1600):
        raise ValueError('height and width must be in 64..1600')
    if source.resolve() == output.resolve():
        raise ValueError("output must differ from the source model")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'unsupported detector SHA-256 {digest}')
    model = onnx.load_model_from_string(raw)
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    slices = [node for node in model.graph.node if node.op_type == 'Slice']
    if len(slices) != 12:
        raise ValueError('expected twelve U-Net skip-connection crops')
    for i, node in enumerate(slices):
        axis = 2 + i % 2
        factor = 2 ** (5 - i // 2)
        old = (800 if axis == 2 else 600) // factor
        ends = constants[node.input[2]]
        if numpy_helper.to_array(ends).tolist() != [old]:
            raise ValueError('unexpected U-Net crop limit')
        if numpy_helper.to_array(constants[node.input[3]]).tolist() != [axis]:
            raise ValueError('unexpected U-Net crop axis')
        data = numpy_helper.to_array(ends).copy()
        data[0] = (height if axis == 2 else width) // factor
        ends.CopyFrom(numpy_helper.from_array(data, ends.name))
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[2].dim_value = height
        value.type.tensor_type.shape.dim[3].dim_value = width
    # Inferred intermediate dimensions from the original export no longer apply.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {'source_sha256': digest, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'height': height, 'width': width, 'source': str(source), 'output': str(output),
            'onnx_version': onnx.__version__, 'operation': 'fixed-shape specialization; weights unchanged'}


def main():
    """Write one derivative and a provenance sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('models/text-detection.onnx'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    args = parser.parse_args()
    manifest = specialize(args.source, args.output, args.height, args.width)
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
