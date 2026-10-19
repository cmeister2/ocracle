#!/usr/bin/env python3
"""Benchmark actual Rust recognition paths on the frozen poster-title fields."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        current = [index]
        for other, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (a != b)))
        previous = current
    return previous[-1]


def comparison(expected, observed):
    expected, observed = normalized(expected), normalized(observed)
    edits = distance(expected, observed)
    return {"exact": expected == observed, "casefold_exact_diagnostic": expected.casefold() == observed.casefold(), "character_edits": edits, "reference_characters": len(expected), "cer": edits / len(expected)}


def historical_models(labels):
    """Resolve archived artifacts; labels are local report keys, not live profiles."""
    retained = json.loads((ROOT / "models/manifest.json").read_text())
    definitions = {}
    for label in labels:
        if label in {"en", "en-medium"}:
            manifest = "paddle-v6-medium-manifest.json" if label == "en-medium" else "paddle-v6-manifest.json"
            declaration = json.loads((ROOT / "models" / manifest).read_text())
            matches = [row for row in declaration["profiles"] if row["id"] == "en" and row["language"] == "en"]
            if len(matches) != 1:
                raise ValueError("Archived model must declare exactly one English artifact")
            definition = {**matches[0], "profile": label}
        elif label == "en-mtg":
            matches = [row for row in retained["adapted_models"] if row["path"] == "models/ocracle-mtg.rten" and row["language"] == "en"]
            if len(matches) != 1:
                raise ValueError("Missing retained English MTG CRNN artifact")
            artifact = matches[0]
            definition = {"model": artifact["path"], "sha256": artifact["sha256"],
                          "configuration": artifact["configuration"], "profile": label,
                          "configuration_sha256": "69a0bb1ed466999a4869d465622b57ff8b2833d51766e21d54ac654c6fa0286e"}
        elif label == "ocrs-base":
            artifact = next(row for row in retained["upstream_models"] if row["path"] == "models/text-recognition-base.rten")
            definition = {"model": artifact["path"], "sha256": artifact["sha256"],
                          "configuration": "training/style-eval/base-config.json",
                          "configuration_sha256": "041498df2c78a559fe8db6d9253e29b5aeadd7f352319eb20622067fd445b1d9"}
        else:
            raise ValueError("Unknown historical model label: " + label)
        config_path = ROOT / definition["configuration"]
        config = json.loads(config_path.read_text())
        if config.get("language") != "en" or digest(config_path) != definition["configuration_sha256"]:
            raise ValueError("Archived English configuration changed: " + label)
        path = ROOT / definition["model"]
        if path.exists() and digest(path) != definition["sha256"]:
            raise ValueError("Present archived model bytes changed: " + label)
        definition["reported"] = {"model_sha256": definition["sha256"],
                                  "configuration_sha256": definition["configuration_sha256"], "configuration": config}
        if "profile" in definition:
            definition["reported"]["profile"] = label
        definitions[label] = definition
    detector = next(row for row in retained["upstream_models"] if row["path"] == "models/text-detection.onnx")
    return definitions, detector


def finite_scores(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "confidence" and (not isinstance(child, (int, float)) or isinstance(child, bool)
                                        or not math.isfinite(child) or not 0 <= child <= 1):
                raise ValueError("Invalid uncalibrated posterior score")
            finite_scores(child)
    elif isinstance(value, list):
        for child in value:
            finite_scores(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite inference value")


def verify_results(path, detected, medium=False):
    """Check complete case membership and metrics against retained raw inference."""
    report = json.loads(path.read_text())
    samples = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()]
    cases = {sample["id"] + "-" + sample["severity"]: sample for sample in samples}
    models = {"en-medium"} if medium else {"en"} if detected else {"en", "en-mtg", "ocrs-base"}
    if report["manifest_sha256"] != digest(DIRECTORY / "manifest.jsonl") or report["provenance_sha256"] != digest(DIRECTORY / "provenance.json"):
        raise ValueError("Result report refers to different frozen challenge data")
    if set(report["models"]) != models:
        raise ValueError("Result report model membership differs")
    definitions, detector = historical_models(models)
    for label, definition in definitions.items():
        expected_model = definition["reported"].copy()
        if detected:
            expected_model["detector_sha256"] = detector["sha256"]
        if report["models"][label] != expected_model:
            raise ValueError("Result differs from archived model/configuration: " + label)
    # The first frozen forced-line report predates the explicit mode field.
    if report.get("inference_mode", "forced_single_line") != ("detected_text" if detected else "forced_single_line"):
        raise ValueError("Recorded inference mode differs")
    if medium and report["assessment"] != "post_selection_no_tuning":
        raise ValueError("Medium assessment was not post-selection")
    expected_pairs = {(case, model) for case in cases for model in models}
    seen = set()
    for row in report["records"]:
        pair = row["case_id"], row["model"]
        if pair in seen or pair not in expected_pairs:
            raise ValueError("Duplicate or unexpected result observation")
        seen.add(pair)
        sample = cases[row["case_id"]]
        if any(row[field] != sample[field] for field in ["card_id", "oracle_id", "card_name", "severity", "path"]):
            raise ValueError("Result observation identity/style metadata differs")
        if row["expected"] != sample["text"] or row["image_sha256"] != sample["image_sha256"] or digest(DIRECTORY / sample["path"]) != sample["image_sha256"]:
            raise ValueError("Result observation differs from frozen pixels/transcription")
        if row["status"] == "ok":
            inference = json.loads(row["stdout"])
            finite_scores(inference)
            if row["returncode"] != 0 or inference != row["inference"] or len(inference["candidates"]) != 2:
                raise ValueError("Retained raw inference/candidates differ")
            model = report["models"][row["model"]]
            if (inference["model"]["sha256"] != model["model_sha256"]
                    or inference["model"]["detector_sha256"] != model.get("detector_sha256")
                    or inference["model"]["language"] != "en"
                    or inference["model"]["name"] != model["configuration"]["model_name"]):
                raise ValueError("Reported recognizer/detector identity differs")
            choices = {"selected_by_confidence": inference, "original": inference["candidates"][0], "enhanced": inference["candidates"][1]}
            _, winner = max(enumerate(inference["candidates"]), key=lambda pair: (pair[1]["confidence"], pair[0]))
            if any(inference[field] != winner[field] for field in ["text", "confidence", "method", "lines"]):
                raise ValueError("Engine selection is not the recorded highest-confidence candidate")
        else:
            if row["status"] != "inference-error" or row["returncode"] == 0:
                raise ValueError("Inference error lacks failing process status")
            choices = {stage: {"text": "", "confidence": 0} for stage in ["selected_by_confidence", "original", "enhanced"]}
        for stage, choice in choices.items():
            result = row["results"][stage]
            if result["text"] != choice["text"] or result["confidence"] != choice["confidence"]:
                raise ValueError("Scored result differs from actual model output")
            for key, value in comparison(sample["text"], choice["text"]).items():
                if result[key] != value:
                    raise ValueError("Scored character metric differs: " + key)
    if seen != expected_pairs:
        raise ValueError("Frozen cases/models were omitted from the report")
    for model in models:
        for stage in ["original", "enhanced", "selected_by_confidence"]:
            for severity in ["all", "clean", "severe"]:
                rows = [row for row in report["records"] if row["model"] == model and (severity == "all" or row["severity"] == severity)]
                edits = sum(row["results"][stage]["character_edits"] for row in rows)
                characters = sum(row["results"][stage]["reference_characters"] for row in rows)
                expected = {"observations": len(rows), "exact": sum(row["results"][stage]["exact"] for row in rows), "casefold_exact_diagnostic": sum(row["results"][stage]["casefold_exact_diagnostic"] for row in rows), "character_edits": edits, "reference_characters": characters, "cer": edits / characters, "inference_errors": sum(row["status"] != "ok" for row in rows)}
                if report["summary"][model][stage][severity] != expected:
                    raise ValueError("Aggregate result metrics differ")
    print(json.dumps({"verified_results": str(path), "actual_cli_runs": len(seen), "retained_candidates": 2 * len(seen), "manifest_sha256": report["manifest_sha256"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/ocracle")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detected", action="store_true", help="Paddle only, using the actual detector and text-layout path instead of --line")
    mode.add_argument("--medium", action="store_true", help="Assess the archived medium artifact on every frozen title, without tuning")
    parser.add_argument("--check", action="store_true", help="Verify existing raw results/metrics offline without inference or file writes")
    args = parser.parse_args()
    output = args.output or DIRECTORY / ("results-medium.json" if args.medium else "results-detected.json" if args.detected else "results.json")
    if args.check:
        verify_results(output, args.detected, args.medium)
        return
    labels = ["en-medium"] if args.medium else ["en"] if args.detected else ["en", "en-mtg", "ocrs-base"]
    definitions, archived_detector = historical_models(labels)
    # The active language registry can change independently of this historical
    # benchmark. Verify only its explicit archived artifacts before inference.
    required = [ROOT / definition["model"] for definition in definitions.values()]
    if args.detected:
        required.append(ROOT / archived_detector["path"])
    if any(not path.exists() for path in required):
        raise ValueError("Missing archived inference weights; download pinned artifacts first")
    if args.detected and digest(ROOT / archived_detector["path"]) != archived_detector["sha256"]:
        raise ValueError("Archived detector digest differs")
    integrity = json.dumps({"verified_historical_models": labels, "uses_active_profiles": False})
    samples = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()]
    if len(samples) != 10:
        raise ValueError("The complete ten-observation frozen set is required")
    provenance = json.loads((DIRECTORY / "provenance.json").read_text())
    if digest(DIRECTORY / "manifest.jsonl") != provenance["manifest_sha256"]:
        raise ValueError("Frozen challenge manifest changed")
    records = []
    source_models = {}
    env = {**os.environ, "RAYON_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory(prefix="ocracle-style-eval-") as temporary:
        snapshot = Path(temporary)
        executable = snapshot / "ocracle"
        shutil.copy2(args.executable, executable)
        (snapshot / "models").mkdir()
        for key, profile in definitions.items():
            for field in ["model", "configuration"]:
                destination = snapshot / profile[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / profile[field], destination)
            if (digest(snapshot / profile["model"]) != profile["sha256"]
                    or digest(snapshot / profile["configuration"]) != profile["configuration_sha256"]):
                raise ValueError("Archived artifact changed during snapshot")
            source_models[key] = profile["reported"].copy()
        if args.detected:
            detector = snapshot / "models/text-detection.onnx"
            shutil.copy2(ROOT / archived_detector["path"], detector)
            if digest(detector) != archived_detector["sha256"]:
                raise ValueError("Archived detector changed during snapshot")
            source_models["en"]["detector_sha256"] = digest(detector)
        for sample in samples:
            image = DIRECTORY / sample["path"]
            if digest(image) != sample["image_sha256"]:
                raise ValueError("Frozen crop changed: " + sample["path"])
            for key in source_models:
                command = [str(executable), "recognize", str(image), "--passes", "all"]
                if args.detected:
                    command += ["--layout", "text", "--detection-model", str(detector)]
                else:
                    command += ["--line"]
                definition = definitions[key]
                command += ["--recognition-model", str(snapshot / definition["model"]),
                            "--model-config", str(snapshot / definition["configuration"])]
                process = subprocess.run(command, text=True, capture_output=True, env=env)
                record = {"case_id": sample["id"] + "-" + sample["severity"], "card_id": sample["card_id"], "oracle_id": sample["oracle_id"], "card_name": sample["card_name"], "severity": sample["severity"], "path": sample["path"], "image_sha256": sample["image_sha256"], "expected": sample["text"], "model": key, "missing_alphabet": sorted(set(sample["text"]) - set(source_models[key]["configuration"]["alphabet"])), "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "status": "inference-error" if process.returncode else "ok"}
                if process.returncode == 0:
                    inference = json.loads(process.stdout)
                    if inference["model"]["sha256"] != source_models[key]["model_sha256"] or inference["model"]["language"] != "en":
                        raise ValueError("Actual CLI model provenance differs from frozen snapshot")
                    if args.detected and inference["model"]["detector_sha256"] != source_models[key]["detector_sha256"]:
                        raise ValueError("Actual CLI detector provenance differs from frozen snapshot")
                    if len(inference["candidates"]) != 2:
                        raise ValueError("Expected both original/enhanced candidates")
                    results = {"selected_by_confidence": {"text": inference["text"], "confidence": inference["confidence"], "method": inference["method"], **comparison(sample["text"], inference["text"])}}
                    for stage, candidate in zip(["original", "enhanced"], inference["candidates"]):
                        results[stage] = {"text": candidate["text"], "confidence": candidate["confidence"], "method": candidate["method"], **comparison(sample["text"], candidate["text"])}
                    if any(not math.isfinite(result["confidence"]) or not 0 <= result["confidence"] <= 1 for result in results.values()):
                        raise ValueError("Invalid uncalibrated posterior score")
                    record.update({"inference": inference, "results": results})
                    print(record["case_id"], key, {stage: result["text"] for stage, result in results.items()}, flush=True)
                else:
                    record["results"] = {stage: {"text": "", "confidence": 0, "error": process.stderr, **comparison(sample["text"], "")} for stage in ["selected_by_confidence", "original", "enhanced"]}
                    print(record["case_id"], key, "INFERENCE ERROR", process.stderr, flush=True)
                records.append(record)
        executable_sha = digest(executable)
    summary = {}
    for key in source_models:
        summary[key] = {}
        for stage in ["original", "enhanced", "selected_by_confidence"]:
            summary[key][stage] = {}
            for severity in ["all", "clean", "severe"]:
                subset = [row for row in records if row["model"] == key and (severity == "all" or row["severity"] == severity)]
                edits = sum(row["results"][stage]["character_edits"] for row in subset)
                characters = sum(row["results"][stage]["reference_characters"] for row in subset)
                summary[key][stage][severity] = {"observations": len(subset), "exact": sum(row["results"][stage]["exact"] for row in subset), "casefold_exact_diagnostic": sum(row["results"][stage]["casefold_exact_diagnostic"] for row in subset), "character_edits": edits, "reference_characters": characters, "cer": edits / characters, "inference_errors": sum(row["status"] != "ok" for row in subset)}
    report = {"schema_version": 1, "evaluated_on": "2026-10-19", "status": "frozen_challenge_no_training_or_profile_readiness_claim", "inference_mode": "detected_text" if args.detected else "forced_single_line", "manifest_sha256": digest(DIRECTORY / "manifest.jsonl"), "provenance_sha256": digest(DIRECTORY / "provenance.json"), "executable_sha256": executable_sha, "integrity_verifier_stdout": integrity, "models": source_models, "preprocessing": "Actual CLI --passes all; original and enhanced candidates plus the engine's confidence selection retained. Expected text used only after inference for scoring.", "metrics": "NFC plus collapsed whitespace, preserving case/punctuation; CER can exceed100%. Casefold exact is a separately labelled diagnostic, never primary exact.", "confidence": "Uncalibrated geometric mean of emitted CTC posterior peaks; no correctness probability. Higher confidence need not be more accurate.", "limitations": "All complete title fields are kept in their original curved/warped layout. Rectangular crops retain annotated neighbouring text/mana/artwork. Failure mixes glyph, segmentation and layout limitations. No upper bound on future models' capability; upstream training membership unknown.", "summary": summary, "records": records}
    if args.medium:
        report["assessment"] = "post_selection_no_tuning"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    verify_results(output, args.detected, args.medium)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
