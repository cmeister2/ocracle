"""Corrupted blur evidence must fail without original scans, models or network."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import blur_prepare


class BlurEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "training/blur-corpus"
        shutil.copytree(blur_prepare.DIRECTORY, self.directory,
                        ignore=shutil.ignore_patterns("cache", "__pycache__"))
        self.provenance = json.loads((self.directory / "provenance.json").read_text())
        for source in self.provenance["identity_exclusions"]["sources"]:
            destination = self.root / source["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(blur_prepare.ROOT / source["path"], destination)
        self.records = [json.loads(line) for line in (self.directory / "manifest.jsonl").read_text().splitlines()]
        self.assertFalse((self.directory / "cache").exists())

    def check(self):
        with (patch.object(blur_prepare, "ROOT", self.root),
              patch.object(blur_prepare, "DIRECTORY", self.directory),
              patch("sys.argv", ["blur_prepare.py", "--check"]),
              patch("urllib.request.urlopen", side_effect=AssertionError("Network is forbidden in offline evidence checks")),
              redirect_stdout(io.StringIO())):
            blur_prepare.main()

    def write_records(self, records):
        # Refresh the outer checksum deliberately, so semantic guards rather
        # than the generic manifest digest must reject the changed records.
        data = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records)
        (self.directory / "manifest.jsonl").write_bytes(data)
        provenance = deepcopy(self.provenance)
        provenance["manifest_sha256"] = hashlib.sha256(data).hexdigest()
        (self.directory / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False))

    def test_frozen_evidence_passes_without_source_cache(self):
        self.check()

    def test_duplicate_case_cannot_replace_another_base_line(self):
        records = deepcopy(self.records)
        target = records[0]
        source = next(row for row in records if row["condition"] == target["condition"]
                      and row["base_id"] != target["base_id"])
        replaced_path = target["path"]
        records[0] = {**deepcopy(source), "path": replaced_path}
        shutil.copy2(self.directory / source["path"], self.directory / replaced_path)
        self.write_records(records)
        # Still448 rows,64 per condition and448 unique paths. One base/condition
        # pair is missing and a different pair is repeated with genuine pixels.
        with self.assertRaises(AssertionError):
            self.check()

    def test_reviewed_label_and_source_identity_are_bound(self):
        for field, value in [("text", "invented transcription"),
                             ("oracle_id", "00000000-0000-0000-0000-000000000000"),
                             ("source_image_sha256", "0" * 64),
                             ("image_url", "https://example.invalid/changed-source.jpg")]:
            with self.subTest(field=field):
                records = deepcopy(self.records)
                records[0][field] = value
                self.write_records(records)
                with self.assertRaises(AssertionError):
                    self.check()

    def test_changed_image_bytes_are_rejected(self):
        image = self.directory / self.records[0]["path"]
        # Trailing data need not change decoded pixels; exact artifact bytes
        # must still match the frozen digest.
        image.write_bytes(image.read_bytes() + b"changed artifact bytes")
        with self.assertRaises(AssertionError):
            self.check()


if __name__ == "__main__":
    unittest.main()
