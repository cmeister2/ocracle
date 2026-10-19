"""Portable historical evidence must retain its original raw process bytes."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mtg_assess


class HistoricalAssessment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = json.loads((mtg_assess.HERE/'mtg-export.json').read_text())
        report = json.loads((mtg_assess.HERE/'mtg-development.json').read_text())
        cls.record = report['records'][0]
        cls.case = mtg_assess.sources('development')[0][0]

    def test_relocated_checkout_accepts_unchanged_historical_commands(self):
        root = Path('/a/clean/ci/checkout')
        with patch.multiple(mtg_assess, ROOT=root, HERE=root/'training/experimental_paddle_training'):
            mtg_assess.validate_record(self.record, self.case, self.export)

    def test_arguments_must_share_the_recorded_checkout_root(self):
        record = copy.deepcopy(self.record)
        record['process']['command'][4] = '/other/checkout/training/experimental_paddle_training/'+self.export['onnx']['path']
        with self.assertRaises(ValueError):
            mtg_assess.validate_record(record, self.case, self.export)

    def test_english_italic_category_survives_import(self):
        cases = mtg_assess.sources('gallery')[0]
        italic = [r for r in cases if r['cohort'] == 'english-gallery' and r['italic']]
        self.assertEqual(len(italic), 8)
        self.assertTrue(any(r['id'] == 'historic-flavor-clean' for r in italic))


if __name__ == '__main__':
    unittest.main()
