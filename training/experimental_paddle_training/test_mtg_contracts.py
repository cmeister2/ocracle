"""Preparation/augmentation guards for the real MTG pilot, without training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import mtg_prepare as prepare
from mtg_augment import CONDITIONS, degrade


class MtgContractTests(unittest.TestCase):
    def test_all_augmentations_preserve_rgb_extent_and_replay(self):
        values = np.random.default_rng(42).integers(0, 256, (31, 179, 3), dtype=np.uint8)
        image = Image.fromarray(values)
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                first, a = degrade(image, condition, 42)
                second, b = degrade(image, condition, 42)
                self.assertEqual(first.mode, 'RGB')
                self.assertEqual(first.size, image.size)
                self.assertEqual(a, b)
                np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            degrade(image, 'unknown', 42)

    def test_motion_does_not_wrap_colours_or_change_constant_regions(self):
        image = Image.new('RGB', (100, 30), (20, 100, 230))
        motion, _ = degrade(image, 'motion', 7)
        np.testing.assert_array_equal(motion, image)

    def test_source_annotation_drift_and_teacher_gold_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = root/'image.bin'
            pixels.write_bytes(b'fixed-input-pixels')
            annotation = root/'reviewed.json'
            annotation.write_text('[]')
            manifest, summary = root/'manifest.jsonl', root/'summary.json'
            row = {'id': 'anchor', 'role': 'teacher-anchor', 'origin': 'english-real-scan',
                   'path': 'image.bin', 'image_sha256': prepare.digest(pixels)}

            def freeze():
                manifest.write_text(json.dumps(row)+'\n')
                summary.write_text(json.dumps({'manifest_sha256': prepare.digest(manifest), 'records': 1,
                    'fonts': [], 'input_sha256': {'reviewed.json': prepare.digest(annotation)},
                    'implementation_sha256': prepare.digest(Path(prepare.__file__)), 'fresh_validation': None}))

            freeze()
            with patch.multiple(prepare, HERE=root, ROOT=root, MANIFEST=manifest, SUMMARY=summary), patch.object(prepare, 'reserved', return_value=(set(), set())):
                prepare.check()
                annotation.write_text('["label changed"]')
                with self.assertRaisesRegex(RuntimeError, 'annotation/provenance changed'):
                    prepare.check()
                annotation.write_text('[]')
                row['text'] = 'normalized old title'
                freeze()
                with self.assertRaisesRegex(RuntimeError, 'normalized gold label'):
                    prepare.check()


if __name__ == '__main__':
    unittest.main()
