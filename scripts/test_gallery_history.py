"""Historical gallery runs must survive default changes without losing identity checks."""
from base64 import b64decode
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gallery_export


class HistoricalGallery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(gallery_export, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.profiles = {}
        for size in ["small", "medium"]:
            configuration = f"models/{size}/fr.json"
            self.write(configuration, {"language": "fr", "alphabet": "Été ", "model_name": size})
            self.profiles[size] = {
                "id": "fr", "language": "fr", "model": f"models/{size}.onnx",
                "sha256": hashlib.sha256(size.encode()).hexdigest(),
                "configuration": configuration,
                "configuration_sha256": gallery_export.sha256(self.root / configuration),
            }
            manifest = "models/paddle-v6-manifest.json" if size == "small" else "models/paddle-v6-medium-manifest.json"
            self.write(manifest, {"schema_version": 1,
                                 "upstream_models": [{"path": self.profiles[size]["model"],
                                                      "sha256": self.profiles[size]["sha256"]}],
                                 "profiles": [self.profiles[size]]})
        self.write("models/manifest.json", {"schema_version": 1, "upstream_models": []})
        # The mutable alias already points to medium; the historical run is small.
        self.write("models/profiles.json", {"schema_version": 1, "profiles": [self.profiles["medium"]]})
        image = self.root / "assets/title.png"
        image.parent.mkdir()
        image.write_bytes(b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="))
        self.fixtures = self.write("assets/fixtures.json", {
            "covered_languages": ["fr"], "samples": [{"id": "title", "language": "fr",
                "source": {"language": "fr"}, "ground_truth": "Été",
                "variants": [{"id": "clean", "image": "title.png"}]}],
        })
        profile = self.profiles["small"]
        candidate = {"text": "Été", "confidence": .9, "method": "original / test fixture"}
        self.result = {
            "recognition_model_sha256": profile["sha256"],
            "fixture_manifest_sha256": gallery_export.sha256(self.fixtures),
            "profile_configurations": {"fr": {"profile_id": "fr", "path": profile["configuration"],
                                               "sha256": profile["configuration_sha256"]}},
            "observations": 1, "exact_matches": 1, "character_error_rate": 0,
            "records": [{"sample_id": "title", "variant_id": "clean",
                "image_sha256": gallery_export.sha256(image), "character_edits": 0,
                "reference_characters": 3, "character_error_rate": 0, "exact_match": True,
                "profile_configuration_sha256": profile["configuration_sha256"],
                "inference": {**candidate, "model": {"sha256": profile["sha256"], "language": "fr",
                    "configuration_sha256": profile["configuration_sha256"]}, "candidates": [candidate]} }],
        }

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))
        return path

    def check(self, result=None):
        path = self.write("assets/results.json", self.result if result is None else result)
        with redirect_stdout(io.StringIO()):
            gallery_export.check_artifacts(self.fixtures, path)

    def test_historical_result_survives_active_alias_repoint(self):
        self.check()
        (self.root / "models/profiles.json").unlink()
        self.check()  # Historical verification does not require the active registry.

    def test_mismatched_model_cannot_reuse_a_historical_profile(self):
        result = deepcopy(self.result)
        result["recognition_model_sha256"] = self.profiles["medium"]["sha256"]
        result["records"][0]["inference"]["model"]["sha256"] = result["recognition_model_sha256"]
        with self.assertRaisesRegex(ValueError, "no matching pinned"):
            self.check(result)

    def test_mismatched_configuration_is_rejected(self):
        result = deepcopy(self.result)
        result["profile_configurations"]["fr"]["path"] = self.profiles["medium"]["configuration"]
        with self.assertRaisesRegex(ValueError, "no matching pinned"):
            self.check(result)
        self.write(self.profiles["small"]["configuration"], {"language": "fr", "alphabet": "téÉ "})
        with self.assertRaisesRegex(ValueError, "Historical profile configuration changed"):
            self.check()

    def test_wrong_declared_print_language_is_rejected(self):
        manifest = json.loads((self.root / "models/paddle-v6-manifest.json").read_text())
        manifest["profiles"][0]["language"] = "de"
        self.write("models/paddle-v6-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "no matching pinned"):
            self.check()

    def test_unknown_historical_profile_is_rejected(self):
        result = deepcopy(self.result)
        result["profile_configurations"]["fr"]["profile_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "no matching pinned"):
            self.check(result)

    def test_optional_medium_manifest_can_declare_new_evidence(self):
        profile = self.profiles["medium"]
        result = deepcopy(self.result)
        result["recognition_model_sha256"] = profile["sha256"]
        recorded = result["profile_configurations"]["fr"]
        recorded.update(path=profile["configuration"], sha256=profile["configuration_sha256"])
        record = result["records"][0]
        record["profile_configuration_sha256"] = profile["configuration_sha256"]
        record["inference"]["model"].update(sha256=profile["sha256"], configuration_sha256=profile["configuration_sha256"])
        self.check(result)


if __name__ == "__main__":
    unittest.main()
