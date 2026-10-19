"""Read-only corruption probes for the stdlib-only frozen artifact checker."""
import hashlib
import json
from pathlib import Path
import unittest
from latin_v2_check import check, InvalidArtifact

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "models/latin-v2-manifest.json"


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


class FrozenV2ArtifactContracts(unittest.TestCase):
    def altered_json(self, path, mutate, pin=None):
        value = json.loads((ROOT / path).read_bytes())
        mutate(value)
        data = encoded(value)
        overrides = {path: data}
        if pin:
            manifest = json.loads((ROOT / MANIFEST).read_bytes())
            manifest[pin] = hashlib.sha256(data).hexdigest()
            overrides[MANIFEST] = encoded(manifest)
        return overrides

    def test_frozen_artifacts_validate_without_training_dependencies(self):
        result = check(ROOT)
        self.assertEqual(result["configurations"], 6)
        self.assertEqual((result["accent_cases"], result["english_cases"]), (20, 30))

    def test_rejects_corrupted_inference_weights(self):
        path = "models/ocracle-mtg-latin-v2.rten"
        data = (ROOT / path).read_bytes()
        with self.assertRaisesRegex(InvalidArtifact, "SHA-256 mismatch"):
            check(ROOT, {path: bytes([data[0] ^ 1]) + data[1:]})

    def test_rejects_corrupted_initializer(self):
        path = "models/training/latin-v1-source.pt"
        with self.assertRaisesRegex(InvalidArtifact, "SHA-256 mismatch"):
            check(ROOT, {path: b"corrupted initializer"})

    def test_rejects_changed_alphabet_even_with_updated_configuration_hash(self):
        path = "models/latin-v2-profiles/fr.json"
        config = json.loads((ROOT / path).read_bytes())
        config["alphabet"] = config["alphabet"][::-1]
        data = encoded(config)
        manifest = json.loads((ROOT / MANIFEST).read_bytes())
        manifest["configuration_sha256"]["fr"] = hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(InvalidArtifact, "alphabet/model/language"):
            check(ROOT, {path: data, MANIFEST: encoded(manifest)})

    def test_rejects_paddle_output_layout_in_shared_crnn_configuration(self):
        overrides = self.altered_json("models/latin-v2-recognizer-config.json",
                                      lambda config: config.update(output_layout="NTC"))
        with self.assertRaisesRegex(InvalidArtifact, "CRNN output"):
            check(ROOT, overrides)

    def test_rejects_paddle_input_in_shared_crnn_configuration(self):
        overrides = self.altered_json("models/latin-v2-recognizer-config.json",
                                      lambda config: config.update(input={"color_mode": "bgr",
                                          "scale": 1 / 127.5, "offset": -1,
                                          "padding_width": 320, "min_content_width": 1,
                                          "width_rounding": "ceil"}))
        with self.assertRaisesRegex(InvalidArtifact, "CRNN input"):
            check(ROOT, overrides)

    def test_accepts_explicit_equivalent_crnn_defaults(self):
        overrides = self.altered_json("models/latin-v2-recognizer-config.json",
                                      lambda config: config.update(output_layout="TNC",
                                          output_values="log_probabilities",
                                          input={"color_mode": "grayscale", "scale": 0.003921568859368563,
                                                 "offset": -.5, "padding_width": 0,
                                                 "min_content_width": 16, "width_rounding": "nearest_even"}))
        self.assertEqual(check(ROOT, overrides)["configurations"], 6)

    def test_rejects_report_step_drift_even_with_updated_report_hash(self):
        overrides = self.altered_json("training/latin-v2-report.json",
                                      lambda report: report.update(selected_step=200),
                                      "training_report_sha256")
        with self.assertRaisesRegex(InvalidArtifact, "step mismatch"):
            check(ROOT, overrides)

    def test_rejects_replay_dataset_drift(self):
        overrides = self.altered_json("training/latin-v2-reproduction-validation.json",
                                      lambda replay: replay.update(manifest_sha256="0" * 64))
        with self.assertRaisesRegex(InvalidArtifact, "dataset binding"):
            check(ROOT, overrides)

    def test_rejects_parity_for_different_weights(self):
        overrides = self.altered_json("training/latin-v2-export-validation-fr.json",
                                      lambda report: report.update(model_sha256="0" * 64))
        with self.assertRaisesRegex(InvalidArtifact, "Parity model mismatch"):
            check(ROOT, overrides)

    def test_rejects_copying_english_parity_over_a_foreign_language_report(self):
        overrides = {"training/latin-v2-export-validation-fr.json":
                     (ROOT / "training/latin-v2-export-validation-en.json").read_bytes()}
        with self.assertRaisesRegex(InvalidArtifact, "fixture language/held-out provenance"):
            check(ROOT, overrides)

    def test_rejects_recomputed_report_hash_with_false_evaluation_metrics(self):
        def mutate(evaluation):
            evaluation["records"][0]["character_edits"] += 1
        overrides = self.altered_json("training/latin-eval/results-v2.json", mutate,
                                      "independent_physical_evaluation_sha256")
        with self.assertRaisesRegex(InvalidArtifact, "per-record metrics"):
            check(ROOT, overrides)

    def test_rejects_false_english_group_denominator_after_rehashing(self):
        def mutate(evaluation):
            evaluation["groups"]["overall"]["reference_characters"] += 1
        overrides = self.altered_json("training/latin-v2-english-gallery.json", mutate,
                                      "english_font_evaluation_sha256")
        with self.assertRaisesRegex(InvalidArtifact, "overall/reference_characters"):
            check(ROOT, overrides)

    def test_rejects_registering_experimental_weights(self):
        manifest = json.loads((ROOT / MANIFEST).read_bytes())
        def mutate(registry):
            registry["profiles"].append({"id": "v2-experiment", "language": "fr",
                                         "model": manifest["path"], "sha256": manifest["sha256"],
                                         "configuration": manifest["configurations"]["fr"]})
        overrides = self.altered_json("models/profiles.json", mutate)
        with self.assertRaisesRegex(InvalidArtifact, "must remain unregistered"):
            check(ROOT, overrides)


if __name__ == "__main__":
    unittest.main()
