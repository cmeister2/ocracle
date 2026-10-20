"""Geometry, pixel and decoder tests; no neural inference or downloads.

Optional PADDLEX_REFERENCE_DIR enables exact comparisons with hash-pinned
upstream Python sources. See mobile_ocr_probe.md for the fetch commands.
"""
import ast
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import types
import typing
import unittest

import cv2
import numpy as np
import pyclipper

import mobile_ocr_probe as probe


DETECTOR = {
    "Global": {"model_name": "test"},
    "PreProcess": {"transform_ops": [
        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
        {"DetResizeForTest": {"resize_long": 960}},
        {"NormalizeImage": {"scale": "1./255.", "order": "hwc",
                            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
    ]},
    "PostProcess": {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.6,
                    "unclip_ratio": 1.5, "max_candidates": 1000},
}


class MobileContracts(unittest.TestCase):
    def test_bgr_order_and_per_channel_normalization(self):
        image = np.full((32, 32, 3), [10, 50, 200], dtype=np.uint8)
        actual = probe.detector_input(image, DETECTOR, 960)
        self.assertEqual(actual.shape, (1, 3, 32, 32))
        for channel, (value, mean, std) in enumerate(zip(
            [10, 50, 200], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], strict=True
        )):
            expected = np.float32(value) * np.float32((1 / 255) / std)
            expected += np.float32(-mean / std)
            np.testing.assert_array_equal(actual[0, channel], expected)

    def test_max_side_rounding_and_tiny_black_padding(self):
        for max_side, expected in [(960, (960, 672)), (640, (640, 448)), (512, (512, 352))]:
            image = np.zeros((1040, 745, 3), np.uint8)
            self.assertEqual(probe.detector_input(image, DETECTOR, max_side).shape[2:], expected)
        tiny = np.full((8, 8, 3), 255, np.uint8)
        result = probe.detector_input(tiny, DETECTOR, 960)
        self.assertEqual(result.shape, (1, 3, 32, 32))
        self.assertGreater(float(result[0, 0, 0, 0]), 0.)
        self.assertLess(float(result[0, 0, 31, 31]), 0.)

    def test_recognition_padding_and_width_cap(self):
        image = np.full((8, 16, 3), [255, 0, 128], np.uint8)
        actual = probe.recognition_input(image, [3, 48, 320])
        self.assertEqual(actual.shape, (3, 48, 320))
        np.testing.assert_array_equal(actual[0, :, :96], 1.)
        np.testing.assert_array_equal(actual[1, :, :96], -1.)
        np.testing.assert_array_equal(actual[:, :, 96:], 0.)
        wide = np.zeros((2, 200, 3), np.uint8)
        self.assertEqual(probe.recognition_input(wide, [3, 48, 320]).shape, (3, 48, 3200))

    def test_ctc_keeps_official_first_ties_and_first_emission_scores(self):
        probabilities = np.array([[
            [0.1, 0.6, 0.3], [0.05, 0.9, 0.05], [1., 0., 0.],
            [0.1, 0.7, 0.2], [0.1, 0.1, 0.8],
        ]], dtype=np.float32)
        text, confidence = probe.ctc_decode(probabilities, ["blank", "é", "龍"])[0]
        self.assertEqual(text, "éé龍")
        self.assertAlmostEqual(confidence, 0.7, places=6)
        tied = np.array([[[0., 0.5, 0.5]]], dtype=np.float32)
        self.assertEqual(probe.ctc_decode(tied, ["blank", "é", "龍"])[0][0], "é")
        self.assertEqual(probe.ctc_decode(np.array([[[1., 0.]]]), ["blank", "a"])[0], ("", 0.))
        with self.assertRaisesRegex(ValueError, "alphabet"):
            probe.ctc_decode(probabilities, ["blank"])

    def test_db_scores_unclipping_and_original_coordinates(self):
        probabilities = np.zeros((64, 96), np.float32)
        probabilities[20:30, 20:60] = 0.9
        boxes = probe.db_boxes(probabilities, (128, 192), DETECTOR)
        self.assertEqual(len(boxes), 1)
        box, score = boxes[0]
        self.assertAlmostEqual(score, 0.9, places=6)
        self.assertLess(box[:, 0].min(), 40)
        self.assertGreater(box[:, 0].max(), 118)
        self.assertTrue(np.all(box >= 0))
        self.assertTrue(np.all(box[:, 0] <= 192))
        self.assertTrue(np.all(box[:, 1] <= 128))
        probabilities[20:30, 20:60] = 0.59
        self.assertEqual(probe.db_boxes(probabilities, (128, 192), DETECTOR), [])

    def test_quad_sort_and_perspective_crop(self):
        def quad(x, y, width=40, height=20):
            return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], np.int32)
        ordered = probe.sort_boxes([(quad(50, 20), 0.8), (quad(10, 25), 0.9), (quad(0, 60), 0.7)])
        self.assertEqual([int(item[0][0, 0]) for item in ordered], [10, 50, 0])
        image = np.full((100, 100, 3), [10, 20, 30], np.uint8)
        crop = probe.crop_quad(image, quad(10, 20))
        self.assertEqual(crop.shape, (20, 40, 3))
        np.testing.assert_array_equal(crop, np.full(crop.shape, [10, 20, 30], np.uint8))
        self.assertEqual(probe.crop_quad(image, quad(10, 20, 20, 40)).shape, (20, 40, 3))

    def test_official_config_sidecar_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            official = directory / "official.json"
            official.write_text(json.dumps(DETECTOR))
            sidecar = directory / "harness.json"
            sidecar.write_text(json.dumps({
                "official_configuration": {"path": "official.json", "sha256": probe.sha256(official)},
            }))
            self.assertEqual(probe.official_config(sidecar)[0], DETECTOR)
            official.write_text(official.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                probe.official_config(sidecar)

    def test_line_calls_run_fresh_recognition_and_report_original_bounds(self):
        class FakeSession:
            """Deterministic model stub, not a neural execution or timing test."""
            def __init__(self):
                self.calls = []

            def get_inputs(self):
                return [types.SimpleNamespace(name="x")]

            def run(self, _, inputs):
                self.calls.append(inputs["x"].copy())
                return [np.array([[[0., 1., 0.], [1., 0., 0.]]], np.float32)]

        engine = probe.MobileOcr.__new__(probe.MobileOcr)
        engine.recognizer = FakeSession()
        engine.characters = ["blank", "é", " "]
        engine.rec_shape = [3, 48, 320]
        engine.batch_size = 6
        image = np.full((20, 100, 3), 255, np.uint8)
        for _ in range(2):
            result = engine.read(image, single_line=True, layout="text")
            self.assertEqual(result["text"], "é")
            self.assertEqual(result["lines"][0]["bounds"], [0, 0, 100, 20])
            self.assertEqual(result["recognition_batches"], [[1, 3, 48, 320]])
            self.assertIsNone(result["detector_input_shape"])
        self.assertEqual(len(engine.recognizer.calls), 2)


REFERENCE_DIR = os.environ.get("PADDLEX_REFERENCE_DIR")


def reference_module(filename, expected_sha256, names, methods=None):
    """Load only selected upstream definitions after verifying source bytes."""
    path = Path(REFERENCE_DIR) / filename
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Unpinned reference source: {path}")
    tree = ast.parse(raw)
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            node.decorator_list = []
            if isinstance(node, ast.ClassDef) and methods:
                node.body = [
                    method for method in node.body
                    if isinstance(method, ast.FunctionDef) and method.name in methods
                ]
            definitions.append(node)
    module = types.ModuleType(filename)
    module.__dict__.update({
        "np": np, "cv2": cv2, "pyclipper": pyclipper, "math": math, "re": re,
        "logging": logging, "Union": typing.Union, "Optional": typing.Optional,
        "List": typing.List, "BaseOperator": object,
    })
    exec(compile(ast.Module(definitions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@unittest.skipUnless(REFERENCE_DIR, "set PADDLEX_REFERENCE_DIR for pinned upstream parity")
class UpstreamParity(unittest.TestCase):
    """Compare preprocessing/geometry against independently pinned PaddleX code."""

    @classmethod
    def setUpClass(cls):
        # Hashes identify exact files at PADDLEX_REVISION, not moving branches.
        cls.detector = reference_module(
            "detection.py", "b23f2349e9e3439d748548730443459d7ed43a4f5e5beb04903f3e5ad58be7c1",
            {"DetResizeForTest", "NormalizeImage", "DBPostProcess"},
        )
        cls.recognizer = reference_module(
            "recognition.py", "772f501de953ecc2f4d6ea7ff3b12500a7766c90dccac740b36de2b6f7cb577e",
            {"validate_text_rec_image_array", "OCRReisizeNormImg", "BaseRecLabelDecode", "CTCLabelDecode"},
        )
        cls.cropper = reference_module(
            "crop.py", "46610b50f06b2d5e0969a9c0f3f71bbbddcc53e4e43eaa1d49811ef94cf0d0a5",
            {"CropByPolys"}, {"__init__", "get_minarea_rect_crop", "get_rotate_crop_image"},
        )

    def test_detector_pixel_parity(self):
        rng = np.random.default_rng(42)
        for shape in [(8, 8, 3), (127, 333, 3), (1040, 745, 3)]:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            for max_side in [512, 960]:
                resized, _ = self.detector.DetResizeForTest(
                    limit_side_len=max_side, limit_type="max"
                )([image])
                reference = self.detector.NormalizeImage(
                    scale=1 / 255, mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], order="hwc",
                )(resized)[0].transpose(2, 0, 1)[None]
                np.testing.assert_array_equal(probe.detector_input(image, DETECTOR, max_side), reference)

    def test_db_geometry_parity(self):
        probabilities = np.zeros((128, 192), np.float32)
        cv2.rectangle(probabilities, (10, 10), (85, 30), 0.9, -1)
        cv2.fillPoly(probabilities, [np.array([[70, 60], [150, 45], [156, 73], [76, 90]])], 0.8)
        for parameters in [
            {"thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5, "max_candidates": 1000},
            {"thresh": 0.2, "box_thresh": 0.4, "unclip_ratio": 1.4, "max_candidates": 3000},
        ]:
            reference = self.detector.DBPostProcess(**parameters)
            boxes, scores = reference.process(
                probabilities[None], np.array([745, 1040, 1., 1.]),
                parameters["thresh"], parameters["box_thresh"], parameters["unclip_ratio"],
            )
            actual = probe.db_boxes(probabilities, (745, 1040), {
                "PostProcess": {"name": "DBPostProcess", **parameters},
            })
            np.testing.assert_array_equal([item[0] for item in actual], boxes)
            np.testing.assert_array_equal([item[1] for item in actual], scores)

    def test_recognition_pixels_and_decoder_parity(self):
        rng = np.random.default_rng(4)
        for shape in [(8, 4, 3), (15, 149, 3), (50, 527, 3), (2, 200, 3)]:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            expected = self.recognizer.OCRReisizeNormImg()([image])[0]
            np.testing.assert_array_equal(probe.recognition_input(image, [3, 48, 320]), expected)
        probabilities = rng.random((2, 20, 4), dtype=np.float32)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected = self.recognizer.CTCLabelDecode(character_list=["é", "龍"])([probabilities])
        actual = probe.ctc_decode(probabilities, ["blank", "é", "龍", " "])
        self.assertEqual([item[0] for item in actual], expected[0])
        self.assertEqual([item[1] for item in actual], expected[1])

    def test_perspective_crop_pixel_parity(self):
        rng = np.random.default_rng(19)
        image = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
        reference = self.cropper.CropByPolys()
        for quad in [
            [[20, 10], [120, 10], [120, 35], [20, 35]],
            [[30, 40], [180, 20], [185, 60], [35, 80]],
            [[20, 10], [50, 10], [50, 120], [20, 120]],
            [[0, 0], [240, 0], [240, 40], [0, 40]],
        ]:
            quad = np.asarray(quad, dtype=np.int16)
            expected = reference.get_minarea_rect_crop(image, quad)
            np.testing.assert_array_equal(probe.crop_quad(image, quad), expected)


if __name__ == "__main__":
    unittest.main()
