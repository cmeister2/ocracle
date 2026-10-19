"""Reject falsified confidence selection even when aggregate metrics agree."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parent / "experimental_colour/evaluate.py"
SPEC = importlib.util.spec_from_file_location("colour_evidence", MODULE)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


class ConfidenceEvidenceContracts(unittest.TestCase):
    """Mutation probes bind recorded selection to the real Rust algorithm."""

    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads(EVIDENCE.RESULTS.read_text())

    def report(self):
        return copy.deepcopy(self.frozen)

    def reject(self, report, message):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupted.json"
            path.write_text(json.dumps(report, ensure_ascii=False))
            with self.assertRaisesRegex(ValueError, message):
                EVIDENCE.check(path)

    def test_lower_confidence_selection_rejected_with_recomputed_metrics(self):
        report = self.report()
        row = next(row for row in report["records"] if
                   row["after"]["candidates"][0]["confidence"] != row["after"]["candidates"][1]["confidence"])
        lower = min(row["after"]["candidates"], key=lambda item: item["confidence"])
        for field in ["text", "confidence", "method", "lines"]:
            row["after"][field] = copy.deepcopy(lower[field])
        report["metrics"]["after"] = EVIDENCE.metrics(report["records"], "after")
        self.reject(report, "highest-confidence")

    def test_nan_in_unselected_candidate_rejected(self):
        report = self.report()
        candidates = report["records"][0]["after"]["candidates"]
        min(candidates, key=lambda item: item["confidence"])["confidence"] = float("nan")
        self.reject(report, "Invalid candidate confidence")

    def test_exact_tie_must_select_last_candidate_like_rust(self):
        report = self.report()
        inference = report["records"][0]["after"]
        for candidate in inference["candidates"]:
            candidate["confidence"] = .99
        first = inference["candidates"][0]
        for field in ["text", "confidence", "method", "lines"]:
            inference[field] = copy.deepcopy(first[field])
        report["metrics"]["after"] = EVIDENCE.metrics(report["records"], "after")
        self.reject(report, "highest-confidence")
