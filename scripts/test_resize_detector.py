"""Verify specialization against the downloaded, hash-pinned detector.

Run with ``python -m unittest discover -s scripts -p test_resize_detector.py``.
Requires ONNX and models/text-detection.onnx from training/download_models.py.
No neural inference or network access is performed.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

import resize_detector as resize


SOURCE = Path(__file__).resolve().parents[1] / 'models/text-detection.onnx'


def dimensions(value):
    """Return the concrete dimensions of an inferred ONNX tensor."""
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


class DetectorSpecialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Read the independently pinned graph once for structural comparisons."""
        cls.raw = SOURCE.read_bytes()
        cls.original = onnx.load_model_from_string(cls.raw)
        cls.constants = {
            tensor.name: numpy_helper.to_array(tensor)
            for tensor in cls.original.graph.initializer
        }

    def test_same_size_preserves_every_operation_and_constant(self):
        """A same-size rewrite cannot change inference math or learned weights."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'same.onnx'
            manifest = resize.specialize(SOURCE, output, 800, 600)
            actual = onnx.load(output)
            self.assertEqual(
                [node.SerializeToString() for node in self.original.graph.node],
                [node.SerializeToString() for node in actual.graph.node],
            )
            self.assertEqual(
                [tensor.name for tensor in self.original.graph.initializer],
                [tensor.name for tensor in actual.graph.initializer],
            )
            for tensor in actual.graph.initializer:
                with self.subTest(tensor=tensor.name):
                    value = numpy_helper.to_array(tensor)
                    expected = self.constants[tensor.name]
                    self.assertEqual(value.dtype, expected.dtype)
                    np.testing.assert_array_equal(value, expected)
            for before, after in zip(
                [*self.original.graph.input, *self.original.graph.output],
                [*actual.graph.input, *actual.graph.output],
                strict=True,
            ):
                self.assertEqual(before.SerializeToString(), after.SerializeToString())
            self.assertEqual(manifest['source_sha256'], hashlib.sha256(self.raw).hexdigest())
            self.assertEqual(manifest['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual((manifest['height'], manifest['width']), (800, 600))

    def test_smaller_graph_has_consistent_shapes_and_only_crop_constants_change(self):
        """Shape inference verifies the whole graph, including odd image sizes."""
        crop_names = {
            node.input[2] for node in self.original.graph.node if node.op_type == 'Slice'
        }
        self.assertEqual(len(crop_names), 12)
        for height, width in [(512, 384), (256, 192), (64, 64), (65, 67)]:
            with self.subTest(height=height, width=width), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'nested' / 'small.onnx'
                resize.specialize(SOURCE, output, height, width)
                actual = onnx.load(output)
                self.assertEqual(
                    [node.SerializeToString() for node in self.original.graph.node],
                    [node.SerializeToString() for node in actual.graph.node],
                )
                changed = set()
                for tensor in actual.graph.initializer:
                    value = numpy_helper.to_array(tensor)
                    expected = self.constants[tensor.name]
                    self.assertEqual(value.dtype, expected.dtype)
                    self.assertEqual(value.shape, expected.shape)
                    if not np.array_equal(value, expected):
                        changed.add(tensor.name)
                self.assertEqual(changed, crop_names)
                inferred = onnx.shape_inference.infer_shapes(
                    actual, check_type=True, strict_mode=True, data_prop=True,
                )
                onnx.checker.check_model(inferred, full_check=True)
                for value in [*inferred.graph.input, *inferred.graph.output]:
                    self.assertEqual(dimensions(value), [1, 1, height, width])
                shapes = {
                    value.name: dimensions(value)
                    for value in [
                        *inferred.graph.input,
                        *inferred.graph.output,
                        *inferred.graph.value_info,
                    ]
                }
                for node in inferred.graph.node:
                    if node.op_type == 'Concat':
                        spatial_shapes = [shapes[name][2:] for name in node.input]
                        self.assertTrue(spatial_shapes)
                        self.assertTrue(all(
                            shape == spatial_shapes[0] for shape in spatial_shapes
                        ))

    def test_unpinned_source_is_rejected_without_touching_output(self):
        """Corruption or a different graph must fail before writing a derivative."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'unsupported.onnx'
            output = Path(directory) / 'output.onnx'
            output.write_bytes(b'existing output')
            source.write_bytes(self.raw + b'changed')
            with self.assertRaisesRegex(ValueError, 'unsupported detector SHA-256'):
                resize.specialize(source, output, 512, 384)
            self.assertEqual(output.read_bytes(), b'existing output')

    def test_invalid_sizes_are_rejected_before_reading_or_writing_files(self):
        """Bounds protect the six-level encoder from invalid spatial sizes."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'does-not-exist.onnx'
            output = Path(directory) / 'output.onnx'
            for height, width in [
                (0, 384), (-1, 384), (63, 384), (512, 63),
                (1601, 384), (512, 1601),
            ]:
                with self.subTest(height=height, width=width):
                    with self.assertRaisesRegex(ValueError, '64..1600'):
                        resize.specialize(source, output, height, width)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
