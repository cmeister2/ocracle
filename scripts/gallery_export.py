#!/usr/bin/env python3
"""Run the actual Rust OCR CLI over held-out gallery fixtures.

No prediction or confidence is supplied by the fixture manifest. Every displayed
prediction is copied from the JSON emitted by the specified executable and model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT / "docs/gallery/assets"


def normalized(text: str) -> str:
    """Normalize Unicode and whitespace without removing OCR punctuation errors."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference: str, observed: str) -> int:
    """Return Levenshtein distance with linear memory use."""
    previous = list(range(len(observed) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(observed, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def sha256(path: Path) -> str:
    """Hash an input artifact so exported observations remain traceable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifacts(fixtures_path: Path, output_path: Path) -> None:
    """Verify the site contains complete, internally consistent real-run artifacts."""
    fixtures = json.loads(fixtures_path.read_text())
    result = json.loads(output_path.read_text())
    model_manifest = json.loads((ROOT / "models/manifest.json").read_text())
    known_recognizers = [
        model for model in model_manifest.get("upstream_models", [])
        if not model.get("training_only") and "detection" not in model["path"]
    ] + model_manifest.get("adapted_models", [])
    model_entry = next((model for model in known_recognizers
                        if model["sha256"] == result["recognition_model_sha256"]), None)
    if model_entry is None:
        raise ValueError(f"Recorded recognition model is not registered in models/manifest.json: {output_path}")
    local_model = ROOT / model_entry["path"]
    if local_model.exists() and sha256(local_model) != model_entry["sha256"]:
        raise ValueError(f"Local model artifact differs from its registered digest: {local_model}")
    configuration = model_entry.get("configuration")
    if configuration and (ROOT / configuration).exists():
        if sha256(ROOT / configuration) != result.get("model_config_sha256"):
            raise ValueError(f"Model configuration changed since OCR was recorded: {configuration}")
    fixture_languages = sorted({sample["language"] for sample in fixtures["samples"]})
    if fixture_languages != sorted(fixtures["covered_languages"]):
        raise ValueError("Declared fixture languages differ from the actual samples")
    if result["fixture_manifest_sha256"] != sha256(fixtures_path):
        raise ValueError("Fixture manifest changed since OCR output was recorded")
    expected_records = {
        (sample["id"], variant["id"]): (sample, variant)
        for sample in fixtures["samples"] for variant in sample["variants"]
    }
    records = result["records"]
    actual_keys = [(record["sample_id"], record["variant_id"]) for record in records]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(expected_records):
        raise ValueError("Gallery output must contain exactly one record per fixture variant")
    total_edits = total_characters = exact_matches = 0
    for record in records:
        sample, variant = expected_records[(record["sample_id"], record["variant_id"])]
        if record["image_sha256"] != sha256(fixtures_path.parent / variant["image"]):
            raise ValueError(f"Image changed since OCR output was recorded: {variant['image']}")
        inference = record["inference"]
        if sample["language"] != sample["source"]["language"] or sample["language"] != inference["model"]["language"]:
            raise ValueError(f"Fixture print language and inference model language differ: {sample['id']}")
        if model_entry.get("language") and model_entry["language"] != inference["model"]["language"]:
            raise ValueError("Inference language differs from the registered model language")
        expected = normalized(sample["ground_truth"])
        observed = normalized(inference["text"])
        edits = edit_distance(expected, observed)
        reference_length = len(expected)
        confidence = inference["confidence"]
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid confidence for {variant['image']}")
        if inference["model"]["sha256"] != result["recognition_model_sha256"]:
            raise ValueError("Inference model identity differs from the recorded model artifact")
        if not inference["method"] or not inference["candidates"]:
            raise ValueError("Inference must retain the actual method and preprocessing candidates")
        if not any(candidate["text"] == inference["text"] and math.isclose(candidate["confidence"], confidence)
                   and candidate["method"] == inference["method"] for candidate in inference["candidates"]):
            raise ValueError("Selected inference does not match any retained candidate")
        if (record["character_edits"] != edits or record["reference_characters"] != reference_length
                or not math.isclose(record["character_error_rate"], edits / max(reference_length, 1))
                or record["exact_match"] != (expected == observed)):
            raise ValueError(f"Stored evaluation does not match observed text: {variant['image']}")
        total_edits += edits
        total_characters += reference_length
        exact_matches += expected == observed
    if (result["observations"] != len(records) or result["exact_matches"] != exact_matches
            or not math.isclose(result["character_error_rate"], total_edits / max(total_characters, 1))):
        raise ValueError("Aggregate gallery metrics do not match the inference records")
    print(f"Verified {len(records)} recorded observations, source hashes, registered model identity, print language and CER metrics")


def main() -> None:
    """Export observed OCR output, confidence, provenance and measured errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/ocracle")
    parser.add_argument("--recognition-model", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify checked-in artifacts without loading models")
    parser.add_argument("--detection-model", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_ASSETS / "fixtures.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ASSETS / "results.json")
    parser.add_argument("--passes", choices=["original", "enhanced", "all"], default="all")
    args = parser.parse_args()
    if args.check:
        outputs = sorted(set([args.output, *args.output.parent.glob("results-*.json")]))
        for output in outputs:
            check_artifacts(args.fixtures, output)
        return
    if args.recognition_model is None:
        parser.error("--recognition-model is required unless --check is used")
    fixtures = json.loads(args.fixtures.read_text())
    fixture_hash = sha256(args.fixtures)
    model_hash = sha256(args.recognition_model)
    executable_hash = sha256(args.executable)
    records = []
    total_edits = 0
    total_characters = 0
    for sample in fixtures["samples"]:
        expected = normalized(sample["ground_truth"])
        for variant in sample["variants"]:
            image_path = args.fixtures.parent / variant["image"]
            command = [str(args.executable.resolve()), "recognize", str(image_path.resolve()),
                       "--recognition-model", str(args.recognition_model.resolve()),
                       "--line", "--passes", args.passes]
            if args.model_config:
                command.extend(["--model-config", str(args.model_config.resolve())])
            if args.detection_model:
                command.extend(["--detection-model", str(args.detection_model.resolve())])
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            prediction = json.loads(process.stdout)
            text = prediction["text"]
            confidence = prediction["confidence"]
            if not isinstance(text, str):
                raise ValueError(f"Non-text prediction for {image_path}")
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence for {image_path}: {confidence!r}")
            observed = normalized(text)
            edits = edit_distance(expected, observed)
            characters = len(expected)
            records.append({
                "sample_id": sample["id"],
                "variant_id": variant["id"],
                "image_sha256": sha256(image_path),
                "character_edits": edits,
                "reference_characters": characters,
                "character_error_rate": edits / max(characters, 1),
                "exact_match": expected == observed,
                "inference": prediction,
            })
            total_edits += edits
            total_characters += characters
            print(f"{sample['id']}/{variant['id']}: {text!r} | confidence {confidence:.4f} | CER {edits / max(characters, 1):.1%}")
    if (fixture_hash != sha256(args.fixtures) or model_hash != sha256(args.recognition_model)
            or executable_hash != sha256(args.executable)):
        raise ValueError("An inference input changed during export; rerun against stable artifacts")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_manifest_sha256": fixture_hash,
        "recognition_model_sha256": model_hash,
        "detection_model_sha256": sha256(args.detection_model) if args.detection_model else None,
        "executable_sha256": executable_hash,
        "model_config_sha256": sha256(args.model_config) if args.model_config else None,
        "passes": args.passes,
        "confidence_note": "Uncalibrated geometric mean of the emitted CTC character posteriors. It is a model score, not the probability that the complete transcription is correct. Selecting among preprocessing passes can increase the score without increasing accuracy.",
        "metric_note": "CER is Levenshtein character edits divided by reference length, after NFC Unicode normalization and collapsing whitespace. Case and punctuation count. CER can exceed 100% when OCR inserts many characters.",
        "character_error_rate": total_edits / max(total_characters, 1),
        "exact_matches": sum(record["exact_match"] for record in records),
        "observations": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    check_artifacts(args.fixtures, temporary)
    temporary.replace(args.output)
    print(f"Wrote {args.output}")
    check_artifacts(args.fixtures, args.output)


if __name__ == "__main__":
    main()
