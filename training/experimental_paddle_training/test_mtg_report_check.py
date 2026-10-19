"""Integrity regressions for the dependency-free frozen pilot checker."""
import copy
import json
from pathlib import Path
import unittest

from mtg_report_check import validate_report

HERE = Path(__file__).resolve().parent


class FrozenPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((HERE/'mtg-pilot.json').read_text())
        cls.summary = json.loads((HERE/'mtg-data.json').read_text())
        cls.records = [json.loads(x) for x in (HERE/'mtg-dataset.jsonl').read_text().splitlines()]

    def test_original_recorded_evidence(self):
        validate_report(self.report, self.records, self.summary)

    def test_corruptions_rejected(self):
        def changed_sample(d):
            d['history'][12]['physical_id'] = d['history'][0]['physical_id']

        def changed_blur(d):
            d['history'][0]['augmentation']['downsample_factor'] = .9

        def changed_target(d):
            d['selection_history'][2]['validation']['examples'][0]['target'] += '.'

        def fabricated_metric(d):
            d['selection_history'][1]['validation']['groups']['all']['edits'] -= 1

        def chosen_test_checkpoint(d):
            d['selected_step'] = 100

        def changed_checkpoint(d):
            d['saved_checkpoints'][1]['sha256'] = '0'*64

        def changed_score(d):
            d['selection_history'][2]['score'] = 0

        def changed_tensor(d):
            d['selection_history'][3]['validation']['examples'][0]['tensor_sha256'] = '0'*64

        def nonfinite_loss(d):
            d['history'][0]['loss'] = float('nan')

        for corrupt in [changed_sample, changed_blur, changed_target, fabricated_metric,
                        chosen_test_checkpoint, changed_checkpoint, changed_score, changed_tensor, nonfinite_loss]:
            with self.subTest(corruption=corrupt.__name__):
                report = copy.deepcopy(self.report)
                corrupt(report)
                with self.assertRaises(ValueError):
                    validate_report(report, self.records, self.summary)

    def test_unsuitable_teacher_labels_rejected(self):
        rows = copy.deepcopy(self.records)
        next(r for r in rows if r['role'] == 'teacher-anchor')['text'] = 'Unsuitable old normalized label'
        with self.assertRaises(ValueError):
            validate_report(self.report, rows, self.summary)


if __name__ == '__main__':
    unittest.main()
