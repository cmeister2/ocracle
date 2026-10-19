#!/usr/bin/env python3
"""Check frozen Latin V2 artifact provenance offline using only the stdlib.

This validates recorded evidence and bytes, not model accuracy through fresh
inference. It never imports PyTorch, downloads files, or changes artifacts.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"en", "fr", "de", "it", "pt", "es"}
MODEL_NAME = "ocracle-mtg-latin-v2"


class InvalidArtifact(ValueError):
    """A frozen artifact or its recorded relationship is inconsistent."""


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


class Reader:
    """Read repository files; optional byte overrides support corruption tests."""

    def __init__(self, root, overrides=None):
        self.root = Path(root).resolve()
        self.overrides = overrides or {}
        self.cache = {}

    def read(self, name):
        relative = Path(name)
        require(not relative.is_absolute(), f"Expected repository-relative path: {name}")
        require((self.root / relative).resolve().is_relative_to(self.root), f"Path escapes repository: {name}")
        key = relative.as_posix()
        if key not in self.cache:
            try:
                self.cache[key] = self.overrides[key] if key in self.overrides else (self.root / relative).read_bytes()
            except OSError as error:
                raise InvalidArtifact(f"Cannot read {name}: {error}") from error
        return self.cache[key]

    def json(self, name):
        try:
            return json.loads(self.read(name))
        except (ValueError, UnicodeError) as error:
            raise InvalidArtifact(f"Invalid JSON in {name}: {error}") from error

    def sha(self, name):
        return hashlib.sha256(self.read(name)).hexdigest()

    def pinned(self, name, expected, size=None):
        require(is_digest(expected), f"Invalid SHA-256 for {name}")
        require(self.sha(name) == expected, f"SHA-256 mismatch: {name}")
        if size is not None:
            require(len(self.read(name)) == size, f"Byte count mismatch: {name}")


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for row, left in enumerate(reference, 1):
        current = [row]
        for column, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"Incorrect {label}")


def check_tensor_contract(config):
    """Mirror the effective CRNN defaults in Rust's ModelConfig/InputConfig."""
    require(config.get("output_layout", "TNC") == "TNC"
            and config.get("output_values", "log_probabilities") == "log_probabilities",
            "CRNN output must be TNC log probabilities")
    settings = config.get("input", {})
    require(isinstance(settings, dict), "CRNN input settings must be an object")
    for key, expected in {"color_mode": "grayscale", "padding_width": 0,
                          "min_content_width": 16, "width_rounding": "nearest_even"}.items():
        value = settings.get(key, expected)
        require(type(value) is type(expected) and value == expected, f"CRNN input contract mismatch: {key}")
    for key, expected in {"scale": 1 / 255, "offset": -.5}.items():
        value = settings.get(key, expected)
        require(type(value) in (int, float) and math.isfinite(value), f"Invalid CRNN input {key}")
        # Rust parses these settings as f32; equivalent serialized f32 values
        # must compare equal to defaults even when their decimal spelling differs.
        try:
            equal = struct.pack("!f", value) == struct.pack("!f", expected)
        except OverflowError:
            equal = False
        require(equal, f"CRNN input contract mismatch: {key}")


def inference(record, language, manifest):
    observed = record["inference"]
    model = observed["model"]
    require(model["sha256"] == manifest["sha256"] and model["name"] == MODEL_NAME
            and model["language"] == language, "Inference model/name/language mismatch")
    confidence = observed["confidence"]
    require(isinstance(confidence, (int, float)) and math.isfinite(confidence)
            and 0 <= confidence <= 1, "Invalid inference confidence")
    candidates = observed["candidates"]
    require(candidates and observed["method"], "Missing inference candidates/method")
    require(any(candidate["text"] == observed["text"]
                and candidate["method"] == observed["method"]
                and math.isclose(candidate["confidence"], confidence)
                for candidate in candidates), "Selected inference is absent from recorded candidates")
    return normalized(observed["text"])


def add_metrics(groups, keys, record, reference, prediction):
    expected = normalized(reference)
    edits = distance(expected, prediction)
    require(record["reference"] == reference, "Recorded reference differs from its source fixture")
    require(record["character_edits"] == edits and record["reference_characters"] == len(expected)
            and record["exact_match"] == (expected == prediction), "Incorrect per-record metrics")
    for key in keys:
        group = groups[key]
        group["observations"] += 1
        group["character_edits"] += edits
        group["reference_characters"] += len(expected)
        group["exact_matches"] += int(expected == prediction)


def check_groups(recorded, derived):
    require(set(recorded) == set(derived), "Evaluation metric groups differ from observations")
    for name, expected in derived.items():
        actual = recorded[name]
        for field, value in expected.items():
            require(actual[field] == value, f"Incorrect {name}/{field}")
        close(actual["character_error_rate"], expected["character_edits"] / max(1, expected["reference_characters"]), f"{name}/CER")


def metric_groups():
    return defaultdict(lambda: {"observations": 0, "character_edits": 0,
                                "reference_characters": 0, "exact_matches": 0})


def check_training_metrics(groups):
    for name, group in groups.items():
        require(group["lines"] > 0 and group["characters"] > 0
                and group["errors"] >= 0 and 0 <= group["exact_lines"] <= group["lines"],
                f"Invalid training metric counts: {name}")
        close(group["cer"], group["errors"] / group["characters"], f"training {name}/CER")
        close(group["exact_line_accuracy"], group["exact_lines"] / group["lines"],
              f"training {name}/exact accuracy")


def check_evaluations(reader, manifest, alphabet):
    """Bind the 20 accent and 30 English observations to their frozen fixtures."""
    accent = reader.json(manifest["independent_physical_evaluation"])
    source_path = "training/latin-eval/manifest.jsonl"
    source_rows = [json.loads(line) for line in reader.read(source_path).splitlines()]
    expected = {(row["card_id"], row["severity"]): row for row in source_rows}
    records = accent["records"]
    keys = [(row["card_id"], row["severity"]) for row in records]
    require(len(expected) == len(source_rows) == len(records) == 20
            and len(set(keys)) == 20 and set(keys) == set(expected), "Accent evaluation must contain all 20 unique fixture cases")
    require(accent["recognition_model_sha256"] == manifest["sha256"]
            and accent["manifest_sha256"] == reader.sha(source_path), "Accent model/fixture manifest mismatch")
    require(is_digest(accent["executable_sha256"]) and accent["passes"] == "all", "Missing accent executable/pass provenance")
    groups = metric_groups()
    for record in records:
        source = expected[(record["card_id"], record["severity"])]
        language = source["language"]
        require(language in LANGUAGES - {"en"} and record["language"] == language
                and record["image"] == source["path"], "Accent fixture language/image mismatch")
        require(record["config_sha256"] == manifest["configuration_sha256"][language], "Accent configuration hash mismatch")
        require(record["image_sha256"] == source["image_sha256"], "Accent image provenance mismatch")
        reader.pinned("training/latin-eval/" + source["path"], source["image_sha256"])
        prediction = inference(record, language, manifest)
        require(record["normalized_prediction"] == prediction, "Incorrect normalized accent prediction")
        require(record["missing_checkpoint_alphabet"] == sorted(set(normalized(source["text"])) - set(alphabet)), "Incorrect missing-alphabet annotation")
        add_metrics(groups, ["overall", "language:" + language, "severity:" + source["severity"]], record, source["text"], prediction)
    check_groups(accent["groups"], groups)

    english = reader.json(manifest["english_font_evaluation"])
    fixture_path = "docs/gallery/assets/fixtures.json"
    fixtures = reader.json(fixture_path)
    expected = {(sample["id"], variant["id"]): (sample, variant)
                for sample in fixtures["samples"] for variant in sample["variants"]}
    records = english["records"]
    keys = [(row["sample_id"], row["variant_id"]) for row in records]
    require(len(expected) == len(records) == len(set(keys)) == 30
            and set(keys) == set(expected), "English evaluation must contain all 30 unique fixture cases")
    require(english["recognition_model_sha256"] == manifest["sha256"]
            and english["fixture_manifest_sha256"] == reader.sha(fixture_path)
            and english["model_config_sha256"] == manifest["configuration_sha256"]["en"], "English model/configuration/fixture mismatch")
    require(is_digest(english["executable_sha256"]) and english["passes"] == "all", "Missing English executable/pass provenance")
    groups = metric_groups()
    for record in records:
        sample, variant = expected[(record["sample_id"], record["variant_id"])]
        require(sample["language"] == sample["source"]["language"] == "en", "Non-English source in English evaluation")
        reader.pinned("docs/gallery/assets/" + variant["image"], record["image_sha256"])
        prediction = inference(record, "en", manifest)
        add_metrics(groups, ["overall", "severity:" + variant["id"]], record, sample["ground_truth"], prediction)
    check_groups(english["groups"], groups)


def check(root=ROOT, overrides=None):
    """Validate committed V2 bytes and cross-file evidence; return a summary."""
    reader = Reader(root, overrides)
    manifest = reader.json("models/latin-v2-manifest.json")
    require(manifest["schema"] == 1 and manifest["model_name"] == MODEL_NAME
            and manifest["path"] == "models/ocracle-mtg-latin-v2.rten", "Unexpected V2 artifact identity")
    require("not registered" in manifest["status"].lower(), "V2 manifest must declare its unregistered status")
    require(set(manifest["languages"]) == LANGUAGES and len(manifest["languages"]) == 6
            and set(manifest["configurations"]) == set(manifest["configuration_sha256"]) == LANGUAGES, "V2 must declare exactly six language configurations")
    reader.pinned(manifest["path"], manifest["sha256"], manifest["bytes"])
    for path_field, hash_field in [("training_report", "training_report_sha256"),
                                  ("independent_physical_evaluation", "independent_physical_evaluation_sha256"),
                                  ("english_font_evaluation", "english_font_evaluation_sha256")]:
        reader.pinned(manifest[path_field], manifest[hash_field])

    corpus_alphabet = reader.json("training/latin-v2-corpus/alphabet.json")
    alphabet = corpus_alphabet["alphabet"]
    require(alphabet and len(set(alphabet)) == len(alphabet)
            and unicodedata.normalize("NFC", alphabet) == alphabet, "Invalid Unicode alphabet")
    shared = reader.json("models/latin-v2-recognizer-config.json")
    require(shared["language"] == "mul-latin", "Shared configuration language mismatch")
    configurations = [shared]
    for language, path in manifest["configurations"].items():
        reader.pinned(path, manifest["configuration_sha256"][language])
        config = reader.json(path)
        require(config["language"] == language, f"Configuration language mismatch: {language}")
        configurations.append(config)
    for config in configurations:
        check_tensor_contract(config)
        require(config["alphabet"] == alphabet and config["input_height"] == 64
                and config["model_name"] == MODEL_NAME and set(config["languages"]) == LANGUAGES
                and len(config["languages"]) == 6, "Configuration alphabet/model/language contract mismatch")
    require(set(corpus_alphabet["languages"]) == LANGUAGES, "Corpus alphabet language mismatch")

    source = reader.json("models/training/latin-v1-source.json")
    require(source == manifest["source_checkpoint"] and source["model_name"] == "ocracle-mtg-latin-v1"
            and source["path"] == "models/training/latin-v1-source.pt", "V1 initializer provenance mismatch")
    reader.pinned(source["path"], source["sha256"], source["bytes"])
    report = reader.json(manifest["training_report"])
    summary = reader.json("training/latin-v2-corpus/summary.json")
    replay = reader.json(manifest["data_replay"])
    require(report["model_name"] == MODEL_NAME and set(report["languages"]) == LANGUAGES
            and report["source_checkpoint"] == source, "Training report model/source mismatch")
    require(report["manifest_sha256"] == summary["manifest_sha256"] == replay["manifest_sha256"]
            == manifest["dataset_manifest_sha256"] and is_digest(manifest["dataset_manifest_sha256"]), "Training/report/replay dataset binding mismatch")
    require(summary["lines"] == replay["replayed_records"] == sum(summary["split"].values())
            == sum(summary["generation"].values()) and summary["alphabet_characters"] == len(alphabet)
            and replay["newly_rendered_images"] == summary["generation"]["latin-v2-synthetic"], "Replay corpus counts/alphabet mismatch")
    require(replay["every_manifest_row_image_hash_verified"] is True
            and replay["full_pipeline_physical_recrop_verified"] is True
            and report["seed"] == summary["seed"] == replay["seed"], "Missing deterministic data replay evidence")
    require(report["steps"] == manifest["training_steps"] > 0
            and report["selected_step"] == manifest["validation_selected_step"]
            and len(report["loss_history"]) == report["steps"], "Training/selected step mismatch")
    history = report["validation_history"]
    steps = [row["step"] for row in history]
    require(steps and steps == sorted(set(steps)) and steps[-1] == report["steps"]
            and all(0 < step <= report["steps"] for step in steps), "Incomplete validation checkpoint history")
    for row in history:
        check_training_metrics(row["groups"])
        physical, english = row["groups"]["new-physical"], row["groups"]["english-preservation"]
        close(row["score"], .75 * physical["cer"] + .15 * (1 - physical["exact_line_accuracy"]) + .1 * english["cer"], "validation selection score")
    selected = min(history, key=lambda row: row["score"])
    require(selected["step"] == report["selected_step"]
            and selected["groups"] == report["v2_validation"]["groups"], "Selected report is not the best fixed validation checkpoint")
    require(report["batch_norm_running_statistics_frozen"] is True
            and manifest["batch_norm_running_statistics_frozen"] is True
            and report["trainable_backbone"] == manifest["trainable_backbone"]
            and report["unchanged_batch_norm_buffers"] > 0
            and set(report["feature_parameter_l2_changes"]) == {"conv.13.weight", "gru.weight_ih_l0"}
            and all(math.isfinite(value) and value > 0 for value in report["feature_parameter_l2_changes"].values()), "Missing fixed-BN/visual training evidence")
    check_training_metrics(report["v2_test"]["groups"])

    require(set(manifest["export_checks"]) == LANGUAGES, "Missing language export checks")
    reviewed = reader.json(manifest["physical_review"])
    english_fixtures = reader.json("docs/gallery/assets/fixtures.json")
    parity_images = {language: {"training/data/latin-v2/" + row["path"]
                                for row in reviewed if row["language"] == language
                                and row["split"] == "test" and row["review_status"] == "accepted"}
                     for language in LANGUAGES - {"en"}}
    parity_images["en"] = {"docs/gallery/assets/" + variant["image"]
                            for sample in english_fixtures["samples"] if sample["language"] == "en"
                            for variant in sample["variants"]}
    for language, path in manifest["export_checks"].items():
        parity = reader.json(path)
        require(parity["model_sha256"] == manifest["sha256"], f"Parity model mismatch: {language}")
        tensors = parity["dynamic_tensor_checks"]
        require({(row["batch"], row["width"]) for row in tensors} == {(1, 64), (1, 273), (2, 160)}
                and len(tensors) == 3, f"Missing dynamic width/batch checks: {language}")
        for row in tensors:
            require(row["shape"] == [row["width"] // 4 + 1, row["batch"], len(alphabet) + 1], "Parity output shape mismatch")
        fixtures = parity["held_out_gallery_checks"]
        require(fixtures, f"Missing actual Rust parity fixture: {language}")
        for row in tensors + fixtures:
            error = row["max_log_probability_error"]
            require(math.isfinite(error) and 0 <= error < .0005, "Parity tolerance exceeded")
        for row in fixtures:
            require(row["image"] in parity_images[language], f"Parity fixture language/held-out provenance mismatch: {language}")
            require(row["pytorch_text"] == row["onnx_text"]
                    and row["pytorch_text"].strip() == row["rust_text"].strip(), "Rust/PyTorch/ONNX transcript mismatch")

    check_evaluations(reader, manifest, alphabet)
    registry = reader.json("models/profiles.json")
    for profile in registry["profiles"]:
        require(profile["sha256"] != manifest["sha256"]
                and profile["model"] != manifest["path"]
                and profile["configuration"] not in manifest["configurations"].values(), "Experimental V2 must remain unregistered")
    return {"model": MODEL_NAME, "sha256": manifest["sha256"], "configurations": 6,
            "accent_cases": 20, "english_cases": 30, "replayed_lines": summary["lines"]}


def main():
    try:
        result = check()
    except (InvalidArtifact, KeyError, TypeError, ValueError) as error:
        print(f"Latin V2 artifact check failed: {error}", file=sys.stderr)
        return 1
    print(f"Verified {result['model']}: frozen weights/source, six configurations/parity reports, "
          f"{result['replayed_lines']} replayed lines, 20 accent +30 English cases; V2 remains unregistered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
