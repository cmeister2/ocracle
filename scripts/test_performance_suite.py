"""Check scoring and pairing rules that could otherwise hide accuracy regressions."""

from copy import deepcopy
import unittest

from performance_report import summaries, validate
from performance_suite import check_inference, score


class PerformanceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "id": "test", "workload": "card", "image_size": [100, 100],
            "regions": [
                {"id": "title", "bounds_xyxy": [0, 0, 80, 20], "text": "Title"},
                {"id": "body", "bounds_xyxy": [0, 40, 100, 80], "text": "Two lines."},
            ],
        }
        self.inference = {
            "image_size": [100, 100], "samples_ms": [3, 1, 2],
            "layout": "standard_card",
            "median_ms": 2, "p95_ms": 3, "text": "Title unscored Two lines.",
            "lines": [
                {"text": "Title", "bounds": [0, 0, 60, 10]},
                {"text": "unscored", "bounds": [0, 25, 100, 10]},
                {"text": "Two", "bounds": [0, 40, 30, 10]},
                {"text": "lines.", "bounds": [0, 60, 60, 10]},
            ],
        }

    def test_card_scoring_counts_missing_lines_and_ignores_only_fixed_regions(self):
        observed = score(self.sample, self.inference)
        self.assertEqual(observed["character_edits"], 0)
        self.assertEqual(observed["unscored_prediction_lines"], 1)
        self.inference["lines"].pop()
        missing = score(self.sample, self.inference)
        self.assertEqual(missing["character_edits"], len(" lines."))
        self.assertFalse(missing["exact_match"])

    def test_punctuation_case_and_accents_are_not_normalized_away(self):
        result = score({"ground_truth": "  Élan — Title.\n"}, {"text": "elan - title"})
        self.assertGreaterEqual(result["character_edits"], 4)
        self.assertEqual(result["reference"], "Élan — Title.")

    def test_out_of_image_coordinates_are_rejected(self):
        check_inference(self.sample, self.inference, 3)
        self.inference["lines"][0]["bounds"] = [0, 0, 101, 10]
        with self.assertRaisesRegex(ValueError, "outside original"):
            check_inference(self.sample, self.inference, 3)

    def test_inconsistent_timing_summary_is_rejected(self):
        self.inference["p95_ms"] = 2
        with self.assertRaisesRegex(ValueError, "p95_ms"):
            check_inference(self.sample, self.inference, 3)

    def test_pairing_rejects_changed_reference_and_keeps_regressions(self):
        record = {"sample": self.sample, "inference": self.inference,
                  "evaluation": score(self.sample, self.inference)}
        before = {"label": "before", "status": "complete", "iterations": 3,
                  "requested_samples": ["test"], "records": [record],
                  "threads": 4, "host": {"machine": "test"}}
        after = deepcopy(before)
        after["label"] = "after"
        after["records"][0]["inference"]["lines"][0]["text"] = "Ttle"
        after["records"][0]["inference"]["text"] = "Ttle unscored Two lines."
        after["records"][0]["evaluation"] = score(
            self.sample, after["records"][0]["inference"])
        identifiers = validate([before, after])
        paired = summaries([before, after], identifiers)["runs"][1]["paired"][0]
        self.assertEqual(paired["character_edits_before"], 0)
        self.assertEqual(paired["character_edits_after"], 1)
        self.assertTrue(paired["text_changed"])
        after["records"][0]["sample"]["regions"][0]["text"] = "Ttle"
        with self.assertRaisesRegex(ValueError, "scoring protocol changed"):
            validate([before, after])

    def test_fake_complete_partial_run_is_rejected(self):
        run = {"label": "partial", "status": "complete", "iterations": 3,
               "requested_samples": ["test", "missing"], "records": [{
                   "sample": self.sample, "inference": self.inference,
                   "evaluation": score(self.sample, self.inference)}],
               "threads": 4, "host": {"machine": "test"}}
        with self.assertRaisesRegex(ValueError, "requested samples"):
            validate([run])

    def test_layout_changes_cannot_be_mistaken_for_optimization(self):
        record = {"sample": self.sample, "inference": self.inference,
                  "evaluation": score(self.sample, self.inference)}
        before = {"label": "before", "status": "complete", "iterations": 3,
                  "requested_samples": ["test"], "records": [record],
                  "threads": 4, "host": {"machine": "test"}}
        after = deepcopy(before)
        after["label"] = "after"
        after["records"][0]["inference"]["layout"] = "text"
        with self.assertRaisesRegex(ValueError, "Paired layout changed"):
            validate([before, after])


if __name__ == "__main__":
    unittest.main()
