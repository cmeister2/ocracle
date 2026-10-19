"""Medium declarations must not weaken frozen small evidence or language routing."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import download_models

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ocracle_model_integrity", ROOT / "scripts/verify_models.py")
integrity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integrity)


class MediumRegistry(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        files = {"models/profiles.json", "models/manifest.json", "models/paddle-v6-manifest.json",
                 "models/LICENSE-PADDLE-APACHE-2.0.txt", "training/experimental_paddle/LICENSE.paddle",
                 "training/experimental_paddle/model-source.json",
                 "training/experimental_paddle/direct-model-config.json",
                 "training/experimental_paddle_candidates/sources.json",
                 "training/experimental_paddle_candidates/sources/v6-medium/inference.yml"}
        registry = json.loads((ROOT / "models/profiles.json").read_text())
        registry["profiles"] = [profile for profile in registry["profiles"]
                                if profile["id"] in integrity.PADDLE_PROFILES | {"en-mtg"}]
        files.update(profile["configuration"] for profile in registry["profiles"])
        small = json.loads((ROOT / "models/paddle-v6-manifest.json").read_text())
        files.update(profile["configuration"] for profile in small["profiles"])
        files.add(small["evaluation"])
        evidence = json.loads((ROOT / small["evaluation"]).read_text())
        files.update(evidence["fixture_manifests"])
        files.update(record["image_path"] for record in evidence["records"])
        for relative in files:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        self.write("models/profiles.json", registry)

    def read(self, path):
        return json.loads((self.root / path).read_text())

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def medium_profile(self, language):
        config = self.read(f"models/paddle-v6/{language}.json")
        config["model_name"] = "Official PP-OCRv6 medium; temporary registry fixture"
        path = f"models/paddle-v6-medium/{language}.json"
        self.write(path, config)
        return {"id": language + "-medium", "language": language, "model": "models/ppocr-v6-medium.onnx",
                "sha256": integrity.MEDIUM_SHA256, "configuration": path,
                "configuration_sha256": hashlib.sha256((self.root / path).read_bytes()).hexdigest()}

    def install_medium(self):
        profiles = [self.medium_profile(language) for language in sorted(integrity.MEDIUM_PROFILE_LANGUAGES.values())]
        manifest = {"schema_version": 1, "name": "PP-OCRv6 medium",
                    "model_path": "models/ppocr-v6-medium.onnx", "sha256": integrity.MEDIUM_SHA256,
                    "bytes": integrity.MEDIUM_BYTES, "source_repository": integrity.MEDIUM_REPOSITORY,
                    "source_revision": integrity.MEDIUM_REVISION,
                    "source_manifest": "training/experimental_paddle_candidates/sources.json",
                    "license": "Apache-2.0", "source_graph_modified": False, "mtg_fine_tuned": False,
                    "upstream_models": [{"url": f"{integrity.MEDIUM_REPOSITORY}/resolve/{integrity.MEDIUM_REVISION}/inference.onnx",
                        "path": "models/ppocr-v6-medium.onnx", "sha256": integrity.MEDIUM_SHA256,
                        "bytes": integrity.MEDIUM_BYTES, "license": "Apache-2.0"}], "profiles": profiles}
        self.write(integrity.MEDIUM_MANIFEST, manifest)
        registry = self.read("models/profiles.json")
        registry["profiles"].extend(profiles)
        self.write("models/profiles.json", registry)
        return manifest

    def test_current_registry_needs_no_medium_manifest_or_downloads(self):
        result = integrity.verify(self.root)
        self.assertEqual((result["profiles"], result["direct_observations"]), (11, 58))
        self.assertFalse((self.root / integrity.MEDIUM_MANIFEST).exists())

    def test_optional_medium_and_latin_aliases_preserve_archived_small_evidence(self):
        archived = (self.root / "models/paddle-v6-manifest.json").read_bytes()
        original_profiles = self.read("models/profiles.json")["profiles"]
        self.install_medium()
        result = integrity.verify(self.root)
        self.assertEqual((result["profiles"], result["direct_observations"]), (17, 58))
        self.assertEqual(self.read("models/profiles.json")["profiles"][:11], original_profiles)
        self.assertIn("models/ppocr-v6-medium.onnx", result["absent_optional_model_files"])
        self.assertEqual((self.root / "models/paddle-v6-manifest.json").read_bytes(), archived)

    def test_archived_small_config_is_checked_after_medium_profiles_are_added(self):
        self.install_medium()
        path = "models/paddle-v6/fr.json"
        config = self.read(path)
        config["alphabet"] = config["alphabet"][1] + config["alphabet"][0] + config["alphabet"][2:]
        self.write(path, config)
        manifest = self.read("models/paddle-v6-manifest.json")
        next(profile for profile in manifest["profiles"] if profile["id"] == "fr")["configuration_sha256"] = integrity.digest(self.root / path)
        self.write("models/paddle-v6-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "character class ordering"):
            integrity.verify(self.root)

    def test_active_alias_requires_the_exact_declared_configuration(self):
        self.install_medium()
        registry = self.read("models/profiles.json")
        profile = next(profile for profile in registry["profiles"] if profile["id"] == "fr-medium")
        original = self.root / profile["configuration"]
        profile["configuration"] = "models/undeclared-fr.json"
        shutil.copy2(original, self.root / profile["configuration"])
        self.write("models/profiles.json", registry)
        with self.assertRaisesRegex(ValueError, "no exact pinned declaration"):
            integrity.verify(self.root)

    def test_unmeasured_languages_cannot_be_declared_for_medium(self):
        manifest = self.install_medium()
        for language in ["la", "ja", "zhs", "zht"]:
            with self.subTest(language=language):
                mutated = deepcopy(manifest)
                mutated["profiles"].append(self.medium_profile(language))
                self.write(integrity.MEDIUM_MANIFEST, mutated)
                with self.assertRaisesRegex(ValueError, "manifest profile set"):
                    integrity.verify(self.root)

    def test_original_alias_cannot_be_repointed_to_medium(self):
        manifest = self.install_medium()
        registry = self.read("models/profiles.json")
        profile = next(profile for profile in manifest["profiles"] if profile["id"] == "fr-medium")
        index = next(index for index, profile in enumerate(registry["profiles"]) if profile["id"] == "fr")
        registry["profiles"][index] = {**profile, "id": "fr"}
        self.write("models/profiles.json", registry)
        with self.assertRaisesRegex(ValueError, "Original profile must retain"):
            integrity.verify(self.root)

    def test_missing_declared_medium_alias_is_rejected(self):
        self.install_medium()
        registry = self.read("models/profiles.json")
        registry["profiles"] = [profile for profile in registry["profiles"] if profile["id"] != "fr-medium"]
        self.write("models/profiles.json", registry)
        with self.assertRaisesRegex(ValueError, "Expected original/optional-medium profile set"):
            integrity.verify(self.root)

    def test_wrong_official_model_and_dictionary_pins_are_rejected(self):
        original = self.install_medium()
        for field, value in [("sha256", "0" * 64), ("source_revision", "0" * 40), ("bytes", 1)]:
            with self.subTest(field=field):
                manifest = deepcopy(original)
                manifest[field] = value
                self.write(integrity.MEDIUM_MANIFEST, manifest)
                with self.assertRaisesRegex(ValueError, "Medium .* changed"):
                    integrity.verify(self.root)
        self.write(integrity.MEDIUM_MANIFEST, original)
        path = original["source_manifest"]
        source = self.read(path)
        candidate = next(candidate for candidate in source["candidates"] if candidate["id"] == "v6-medium")
        candidate["files"]["inference.yml"]["sha256"] = "0" * 64
        self.write(path, source)
        with self.assertRaisesRegex(ValueError, "graph/dictionary pin differs"):
            integrity.verify(self.root)

    def test_changed_medium_tensor_contract_is_rejected_after_rehashing(self):
        manifest = self.install_medium()
        profile = next(profile for profile in manifest["profiles"] if profile["id"] == "fr-medium")
        config = self.read(profile["configuration"])
        config["input"]["color_mode"] = "rgb"
        self.write(profile["configuration"], config)
        profile["configuration_sha256"] = integrity.digest(self.root / profile["configuration"])
        self.write(integrity.MEDIUM_MANIFEST, manifest)
        with self.assertRaisesRegex(ValueError, "Paddle input contract"):
            integrity.verify(self.root)

    def test_present_corrupt_medium_weights_fail_even_in_optional_mode(self):
        manifest = self.install_medium()
        (self.root / manifest["model_path"]).write_bytes(b"corrupted graph")
        with self.assertRaisesRegex(ValueError, "Model SHA-256 mismatch"):
            integrity.verify(self.root)

    def test_downloader_includes_medium_only_when_manifest_exists(self):
        manifest = self.install_medium()
        with (patch.object(download_models, "ROOT", self.root),
              patch.object(download_models, "fetch") as fetch,
              patch("sys.argv", ["download_models.py"])):
            download_models.main()
            medium = manifest["upstream_models"][0]
            fetch.assert_any_call(medium["url"], self.root / medium["path"], medium["sha256"])
            count = fetch.call_count
            (self.root / integrity.MEDIUM_MANIFEST).unlink()
            fetch.reset_mock()
            download_models.main()
            self.assertEqual(fetch.call_count, count - 1)
            self.assertFalse(any(call.args[1] == self.root / medium["path"] for call in fetch.call_args_list))


if __name__ == "__main__":
    unittest.main()
