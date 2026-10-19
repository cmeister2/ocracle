#!/usr/bin/env python3
"""Verify registered model identities and frozen color-CTC evidence offline.

Only the Python 3.12+ standard library is required. Missing downloaded weights are
reported by default; --require-models requires every inference artifact. Any
present artifact must match its pinned digest, including optional checkpoints.
No inference, downloads, model conversion or file mutation occurs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PADDLE_REPOSITORY = "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx"
PADDLE_REVISION = "b8f84f0b80c529de40b4fbb3544b84fa7233a513"
PADDLE_SHA256 = "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
PADDLE_BYTES = 21_159_378
PADDLE_YAML_SHA256 = "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1"
# SHA-256 of the exact UTF-8 character sequence, including the appended space.
# Established against the pinned publisher dictionary, without sorting or NFC
# normalization: Unicode-equivalent characters may occupy distinct class rows.
PADDLE_ALPHABET_SHA256 = "d9a64a84befbb1f2e77ab4f2454acdb9cf1995e96567555aa6c30c2256d0fecb"
MTG_ALPHABET_SHA256 = "1e3585bda6d8ad5f5f29dbafb6a56e21102bd027bbef36c68eca3b6f60dadf1e"
PADDLE_PROFILES = {"en", "es", "fr", "de", "it", "pt", "la", "ja", "zhs", "zht"}
MEDIUM_PROFILE_LANGUAGES = {language: language for language in ("en", "es", "fr", "de", "it", "pt")}
MEDIUM_MANIFEST = "models/paddle-v6-medium-manifest.json"
MEDIUM_REPOSITORY = "https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx"
MEDIUM_REVISION = "50c7eacafc52fa7bcf4194e8cd08e46f8558504b"
MEDIUM_SHA256 = "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"
MEDIUM_BYTES = 76_554_979
MEDIUM_YAML_SHA256 = "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129"
PROFILE_FIELDS = ("id", "language", "model", "sha256", "configuration", "configuration_sha256")
CONTRACT_FIELDS = ("alphabet", "input_height", "output_layout", "output_values")


def require(condition, message):
    """Raise a contextual integrity error; never rely on removable assertions."""
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    """Reject duplicate JSON keys rather than silently retaining the last one."""
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relative_file(root, value):
    """Resolve declared repository paths without permitting external targets."""
    require(isinstance(value, str) and value, "Empty or non-string artifact path")
    candidate = Path(value)
    require(not candidate.is_absolute() and ".." not in candidate.parts, f"Non-repository artifact path: {value}")
    resolved = (root / candidate).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Artifact escapes repository: {value}")
    return resolved


def sha256_value(value, context):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid SHA-256: {context}")


def checked_weight(root, item, required, checked, missing):
    """Check present bytes; track absent downloads distinctly from corruption."""
    path = relative_file(root, item["path"])
    sha256_value(item["sha256"], item["path"])
    if path.exists():
        require(path.is_file(), f"Model artifact is not a file: {item['path']}")
        require(digest(path) == item["sha256"], f"Model SHA-256 mismatch: {item['path']}")
        if "bytes" in item:
            require(path.stat().st_size == item["bytes"], f"Model byte size mismatch: {item['path']}")
        checked.add(item["path"])
    else:
        missing.add(item["path"])
        require(not required, f"Required model is absent: {item['path']}; run training/download_models.py")


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def verify_evidence(root, manifest, configurations):
    """Bind small-model observations to archived contracts and frozen pixels."""
    report = read_json(relative_file(root, manifest["evaluation"]))
    require(report["original_model_sha256"] == PADDLE_SHA256, "Direct evidence identifies different model weights")
    template_path = relative_file(root, "training/experimental_paddle/direct-model-config.json")
    require(digest(template_path) == report["configuration_template_sha256"], "Direct evidence template changed")
    template = read_json(template_path)
    require(len(report["records"]) == 58, "Direct evidence must retain all 58 fixed observations")
    for path, expected in report["fixture_manifests"].items():
        require(digest(relative_file(root, path)) == expected, f"Direct evidence fixture manifest changed: {path}")
    seen = set()
    cohort_counts = {}
    for record in report["records"]:
        require(record["id"] not in seen, f"Duplicate direct observation: {record['id']}")
        seen.add(record["id"])
        config = configurations[record["language"]]
        for key in CONTRACT_FIELDS:
            require(config[key] == template[key], f"Direct evidence contract differs for {record['language']}: {key}")
        for key in ("color_mode", "scale", "offset", "padding_width", "width_rounding"):
            require(config["input"][key] == template["input"][key], f"Direct evidence input differs for {record['language']}: {key}")
        path = relative_file(root, record["image_path"])
        require(digest(path) == record["image_sha256"], f"Direct observation pixels changed: {record['id']}")
        # Historical evidence predates configurable content minima. Prove this
        # distinction cannot change any of its actual crops; narrow-glyph input
        # behavior is covered by the Rust tensor regression instead.
        with path.open("rb") as image:
            header = image.read(24)
        require(header[:8] == b"\x89PNG\r\n\x1a\n", f"Direct fixture is not PNG: {record['id']}")
        width, height = struct.unpack(">II", header[16:24])
        natural_width = math.ceil(width * config["input_height"] / height)
        require(natural_width >= max(template["input"].get("min_content_width", 16), config["input"]["min_content_width"]), f"Direct evidence content minimum changed input: {record['id']}")
        require(record["ort_rten_identical_argmax"] is True, f"Recorded ORT/RTen argmax disagreement: {record['id']}")
        require(record["output_shape"][-1] == len(config["alphabet"]) + 1, f"Recorded CTC classes disagree: {record['id']}")
        require(record["input_shape"][1:3] == [3, 48], f"Recorded model channels/height disagree: {record['id']}")
        expected = normalized(record["expected"])
        require(record["expected_characters"] == len(expected), f"Recorded reference length disagrees: {record['id']}")
        texts = set()
        for name, result in record["results"].items():
            observed = normalized(result["text"])
            require(result["exact"] == (observed == expected), f"Recorded exact metric disagrees: {record['id']} {name}")
            require(result["character_edits"] == distance(expected, observed), f"Recorded CER numerator disagrees: {record['id']} {name}")
            texts.add(result["text"])
        require(len(texts) == 1, f"Recorded direct pipelines disagree: {record['id']}")
        cohort_counts[record["cohort"]] = cohort_counts.get(record["cohort"], 0) + 1
    require(cohort_counts == {"cjk": 8, "latin": 20, "gallery": 30}, "Direct evidence cohort membership changed")
    return len(seen)


def profile_identity(profile):
    """Return the complete runtime identity, excluding descriptive metadata."""
    return tuple(profile[field] for field in PROFILE_FIELDS)


def verify_paddle_profiles(root, manifest, languages, exact_set=False):
    """Verify pinned declarations and each explicit profile-to-language mapping."""
    profiles = {profile["id"]: profile for profile in manifest["profiles"]}
    require(len(profiles) == len(manifest["profiles"]), "Duplicate Paddle manifest profile ID")
    allowed = set(languages)
    require(set(profiles) <= allowed and (not exact_set or set(profiles) == allowed), "Paddle manifest profile set differs")
    configurations = {}
    for key, profile in profiles.items():
        require(profile["language"] == languages[key] and profile["model"] == manifest["model_path"]
                and profile["sha256"] == manifest["sha256"], f"Paddle manifest profile routing mismatch: {key}")
        path = relative_file(root, profile["configuration"])
        require(digest(path) == profile["configuration_sha256"], f"Profile configuration SHA-256 mismatch: {key}")
        config = read_json(path)
        require(config["language"] == languages[key], f"Profile/configuration language mismatch: {key}")
        alphabet = config["alphabet"]
        require(isinstance(alphabet, str) and len(alphabet) == 18709 and len(set(alphabet)) == 18709
                and alphabet.endswith(" "), f"Paddle blank/dictionary/space mapping mismatch: {key}")
        require(hashlib.sha256(alphabet.encode("utf-8")).hexdigest() == PADDLE_ALPHABET_SHA256,
                f"Paddle character class ordering changed: {key}")
        require(config["input_height"] == 48 and config["output_layout"] == "NTC"
                and config["output_values"] == "probabilities", f"Paddle tensor contract mismatch: {key}")
        expected_input = {"color_mode": "bgr", "offset": -1, "padding_width": 320,
                          "width_rounding": "ceil", "min_content_width": 1}
        for field, value in expected_input.items():
            require(config["input"].get(field) == value, f"Paddle input contract mismatch: {key} {field}")
        require(math.isclose(config["input"]["scale"], 2 / 255, rel_tol=0, abs_tol=1e-12), f"Paddle pixel scale changed: {key}")
        require(not set("¡„ẞ") & set(alphabet) and set("ïÄß選択集群攻击劃破蒼穹") <= set(alphabet),
                f"Paddle documented glyph audit changed: {key}")
        configurations[key] = config
    return profiles, configurations


def verify_medium_source(root, manifest):
    """Verify only the immutable official medium graph and publisher dictionary."""
    require(manifest["schema_version"] == 1, "Unsupported medium manifest schema")
    require(manifest["source_repository"] == MEDIUM_REPOSITORY
            and manifest["source_revision"] == MEDIUM_REVISION, "Medium publisher/revision changed")
    require(manifest["sha256"] == MEDIUM_SHA256 and manifest["bytes"] == MEDIUM_BYTES, "Medium model identity changed")
    require(manifest["source_graph_modified"] is False and manifest["mtg_fine_tuned"] is False,
            "Official medium provenance changed")
    require(manifest["license"] == "Apache-2.0", "Medium license declaration changed")
    require(len(manifest["upstream_models"]) == 1 and not manifest.get("adapted_models"),
            "Medium manifest must declare only the original upstream graph")
    require(manifest["source_manifest"] == "training/experimental_paddle_candidates/sources.json", "Medium source manifest changed")
    source_path = relative_file(root, manifest["source_manifest"])
    source = read_json(source_path)
    candidates = [item for item in source["candidates"] if item["id"] == "v6-medium"]
    require(source["schema_version"] == 1 and source["publisher"] == "PaddlePaddle"
            and len(candidates) == 1, "Medium publisher evidence changed")
    candidate = candidates[0]
    require(candidate["repository"] == "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
            and candidate["revision"] == MEDIUM_REVISION and candidate["license"] == "apache-2.0",
            "Medium source evidence revision/license differs")
    for name, sha, size in [("inference.onnx", MEDIUM_SHA256, MEDIUM_BYTES),
                            ("inference.yml", MEDIUM_YAML_SHA256, 150_580)]:
        item = candidate["files"][name]
        require(item["sha256"] == sha and item["bytes"] == size
                and item["url"] == f"{MEDIUM_REPOSITORY}/resolve/{MEDIUM_REVISION}/{name}",
                f"Medium source graph/dictionary pin differs: {name}")
        relative = str(Path(manifest["source_manifest"]).parent / item["path"])
        path = relative_file(root, relative)
        require(name != "inference.yml" or path.is_file(), "Pinned medium dictionary is absent")
        if path.exists():
            require(path.stat().st_size == size and digest(path) == sha, f"Medium source bytes differ: {name}")


def verify_historical_mtg(root, legacy):
    """Retain the CRNN tensor audit without exposing a quality-choice profile."""
    declarations = [item for item in legacy["adapted_models"] if item["path"] == "models/ocracle-mtg.rten"]
    require(len(declarations) == 1, "Retained MTG CRNN declaration changed")
    declaration = declarations[0]
    require(declaration["language"] == "en" and declaration["configuration"] == "models/recognizer-config.json",
            "Retained MTG CRNN configuration routing changed")
    config = read_json(relative_file(root, declaration["configuration"]))
    alphabet = config["alphabet"]
    require(config["language"] == "en" and len(alphabet) == 96
            and hashlib.sha256(alphabet.encode()).hexdigest() == MTG_ALPHABET_SHA256, "Retained MTG CRNN class mapping changed")
    require(config["input_height"] == 64 and config["output_layout"] == "TNC"
            and config.get("output_values", "log_probabilities") == "log_probabilities", "Retained MTG CRNN tensor contract changed")
    require(config.get("blank_index", 0) == 0 and config.get("architecture", "CRNN") == "CRNN",
            "Retained MTG CRNN blank/architecture declaration changed")
    defaults = {"color_mode": "grayscale", "scale": 1 / 255, "offset": -.5, "padding_width": 0,
                "width_rounding": "nearest_even", "min_content_width": 16}
    for field, value in defaults.items():
        require(config.get("input", {}).get(field, value) == value, f"Retained MTG CRNN input contract changed: {field}")


def verify(root, require_models=False):
    """Verify source pinning, dictionaries, profiles, optional weights and evidence."""
    root = root.resolve()
    registry = read_json(root / "models/profiles.json")
    paddle = read_json(root / "models/paddle-v6-manifest.json")
    legacy = read_json(root / "models/manifest.json")
    require(registry["schema_version"] == paddle["schema_version"] == 1, "Unsupported model registry/manifest schema")
    require(paddle["source_repository"] == PADDLE_REPOSITORY and paddle["source_revision"] == PADDLE_REVISION, "Paddle publisher/revision changed")
    require(paddle["sha256"] == PADDLE_SHA256 and paddle["bytes"] == PADDLE_BYTES, "Paddle model identity changed")
    require(paddle["source_graph_modified"] is False and paddle["mtg_fine_tuned"] is False, "Upstream color baseline provenance changed")
    require(paddle["license"] == "Apache-2.0", "Paddle license declaration changed")
    source = read_json(relative_file(root, paddle["source_manifest"]))
    require(source["revision"] == PADDLE_REVISION and source["model_repository"] == PADDLE_REPOSITORY, "Paddle source evidence revision differs")
    require(source["files"]["inference.onnx"]["sha256"] == PADDLE_SHA256, "Source graph digest differs")
    require(source["files"]["inference.yml"]["sha256"] == PADDLE_YAML_SHA256, "Publisher dictionary/config digest differs")
    require(source["files"]["inference.yml"]["url"] == f"{PADDLE_REPOSITORY}/resolve/{PADDLE_REVISION}/inference.yml", "Publisher dictionary/config URL differs")
    require(source["original_contract"]["input"]["shape"] == ["N", 3, 48, "W"] and source["original_contract"]["output"]["shape"] == ["N", "T", 18710], "Publisher tensor shape audit differs")
    for name, expected in (("inference.onnx", PADDLE_SHA256), ("inference.yml", PADDLE_YAML_SHA256)):
        cached = root / "training/experimental_paddle/cache" / name
        if cached.exists():
            require(digest(cached) == expected, f"Cached publisher graph/config SHA-256 mismatch: {name}")
    dictionary = source["original_contract"]["dictionary"]
    require(dictionary["entries"] == dictionary["unique_entries"] == 18708 and dictionary["class_count"] == 18710, "Publisher dictionary/class audit changed")
    require(dictionary["all_entries_single_Unicode_scalar"] is True and dictionary["contains_space"] is False, "Publisher dictionary entry audit changed")
    require(digest(root / "models/LICENSE-PADDLE-APACHE-2.0.txt") == digest(root / "training/experimental_paddle/LICENSE.paddle"), "Distributed Paddle license differs from pinned source license")
    verify_historical_mtg(root, legacy)
    small_profiles, configurations = verify_paddle_profiles(root, paddle, {key: key for key in PADDLE_PROFILES}, exact_set=True)
    medium_profiles = {}
    declarations = [legacy, paddle]
    medium_path = root / MEDIUM_MANIFEST
    if medium_path.exists():
        medium = read_json(medium_path)
        verify_medium_source(root, medium)
        medium_profiles, _ = verify_paddle_profiles(root, medium, MEDIUM_PROFILE_LANGUAGES, exact_set=True)
        declarations.append(medium)
    artifacts = {}
    checked, missing = set(), set()
    for declaration in declarations:
        for item in declaration["upstream_models"]:
            require(item["path"] not in artifacts, f"Duplicate upstream model path: {item['path']}")
            require(item["url"].startswith("https://"), f"Non-HTTPS model source: {item['path']}")
            artifacts[item["path"]] = item
            checked_weight(root, item, require_models and not item.get("training_only", False), checked, missing)
        for item in declaration.get("adapted_models", []):
            require(item["path"] not in artifacts, f"Duplicate adapted model path: {item['path']}")
            artifacts[item["path"]] = item
            checked_weight(root, item, require_models, checked, missing)
    original = artifacts[paddle["model_path"]]
    require(original["sha256"] == PADDLE_SHA256 and original["bytes"] == PADDLE_BYTES, "Downloader Paddle declaration differs")
    require(original["url"] == f"{PADDLE_REPOSITORY}/resolve/{PADDLE_REVISION}/inference.onnx", "Downloader URL is not the pinned original graph")
    require(original["license"] == "Apache-2.0", "Downloader Paddle license differs")
    if medium_path.exists():
        original = artifacts[medium["model_path"]]
        require(original["sha256"] == MEDIUM_SHA256 and original["bytes"] == MEDIUM_BYTES,
                "Downloader medium declaration differs")
        require(original["url"] == f"{MEDIUM_REPOSITORY}/resolve/{MEDIUM_REVISION}/inference.onnx",
                "Downloader medium URL is not the pinned original graph")
        require(original["license"] == "Apache-2.0" and not original.get("training_only", False),
                "Downloader medium license/inference declaration differs")
    profiles = {}
    for profile in registry["profiles"]:
        key = profile["id"]
        require(key not in profiles, f"Duplicate profile ID: {key}")
        profiles[key] = profile
        require(profile["model"] in artifacts, f"Profile has no pinned model declaration: {key}")
        require(profile["sha256"] == artifacts[profile["model"]]["sha256"], f"Profile/model digest mismatch: {key}")
        adapted_config = artifacts[profile["model"]].get("configuration")
        if adapted_config:
            require(profile["configuration"] == adapted_config, f"Adapted model configuration routing mismatch: {key}")
        config_path = relative_file(root, profile["configuration"])
        require(digest(config_path) == profile["configuration_sha256"], f"Profile configuration SHA-256 mismatch: {key}")
        config = read_json(config_path)
        require(config["language"] == profile["language"], f"Profile/configuration language mismatch: {key}")
        alphabet = config["alphabet"]
        require(isinstance(alphabet, str) and len(alphabet) == len(set(alphabet)), f"Invalid/duplicate class alphabet: {key}")
        require(not any(0xD800 <= ord(c) <= 0xDFFF for c in alphabet), f"Non-scalar Unicode class label: {key}")
        require(key in PADDLE_PROFILES, f"Unexpected non-language profile: {key}")
        expected_profile = medium_profiles.get(key, small_profiles[key])
        require(profile_identity(profile) == profile_identity(expected_profile), f"Active language profile has no exact pinned declaration: {key}")
    require(set(profiles) == PADDLE_PROFILES, "Expected exactly ten printed-language profile IDs")
    observations = verify_evidence(root, paddle, configurations)
    return {"profiles": len(profiles), "paddle_character_classes_including_blank": 18710, "direct_observations": observations, "verified_model_files": sorted(checked), "absent_optional_model_files": sorted(missing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (default: this checkout)")
    parser.add_argument("--require-models", action="store_true", help="Require all inference weights, as after download_models.py")
    args = parser.parse_args()
    try:
        result = verify(args.root, args.require_models)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, struct.error) as error:
        print(f"Model integrity verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
