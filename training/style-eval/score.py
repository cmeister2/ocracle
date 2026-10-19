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
import sys
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


def verify_results(path, detected):
    """Check complete case membership and metrics against retained raw inference."""
    report = json.loads(path.read_text())
    samples = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()]
    cases = {sample["id"] + "-" + sample["severity"]: sample for sample in samples}
    models = {"en"} if detected else {"en", "en-mtg", "ocrs-base"}
    if report["manifest_sha256"] != digest(DIRECTORY / "manifest.jsonl") or report["provenance_sha256"] != digest(DIRECTORY / "provenance.json"):
        raise ValueError("Result report refers to different frozen challenge data")
    if set(report["models"]) != models:
        raise ValueError("Result report model membership differs")
    expected_pairs = {(case, model) for case in cases for model in models}
    seen = set()
    for row in report["records"]:
        pair = row["case_id"], row["model"]
        if pair in seen or pair not in expected_pairs:
            raise ValueError("Duplicate or unexpected result observation")
        seen.add(pair)
        sample = cases[row["case_id"]]
        if row["expected"] != sample["text"] or row["image_sha256"] != sample["image_sha256"] or digest(DIRECTORY / sample["path"]) != sample["image_sha256"]:
            raise ValueError("Result observation differs from frozen pixels/transcription")
        if row["status"] == "ok":
            inference = json.loads(row["stdout"])
            if inference != row["inference"] or len(inference["candidates"]) != 2:
                raise ValueError("Retained raw inference/candidates differ")
            model = report["models"][row["model"]]
            if inference["model"]["sha256"] != model["model_sha256"] or inference["model"]["detector_sha256"] != model.get("detector_sha256"):
                raise ValueError("Reported recognizer/detector identity differs")
            choices = {"selected_by_confidence": inference, "original": inference["candidates"][0], "enhanced": inference["candidates"][1]}
            if inference["confidence"] != max(candidate["confidence"] for candidate in inference["candidates"]):
                raise ValueError("Engine selection is not the recorded highest-confidence candidate")
        else:
            if row["returncode"] == 0:
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
    parser.add_argument("--detected", action="store_true", help="Paddle only, using the actual detector and text-layout path instead of --line")
    parser.add_argument("--check", action="store_true", help="Verify existing raw results/metrics offline without inference or file writes")
    args = parser.parse_args()
    output = args.output or DIRECTORY / ("results-detected.json" if args.detected else "results.json")
    if args.check:
        verify_results(output, args.detected)
        return
    # This gate is deliberately offline: source pinning and the actually present
    # production model bytes must verify before the experiment uses the CLI.
    integrity = subprocess.run([sys.executable, str(ROOT / "scripts/verify_models.py"), "--require-models"], text=True, capture_output=True, check=True)
    samples = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()]
    if len(samples) != 10:
        raise ValueError("The complete ten-observation frozen set is required")
    provenance = json.loads((DIRECTORY / "provenance.json").read_text())
    if digest(DIRECTORY / "manifest.jsonl") != provenance["manifest_sha256"]:
        raise ValueError("Frozen challenge manifest changed")
    registry = json.loads((ROOT / "models/profiles.json").read_text())
    profiles = {row["id"]: row for row in registry["profiles"]}
    base_config = json.loads((ROOT / "models/recognizer-config.json").read_text())
    base_config.update({"model_name": "Ocrs generic English CRNN (s52qdbqt)", "architecture": "CRNN", "language": "en", "output_values": "log_probabilities", "input": {"color_mode": "grayscale", "scale": 1 / 255, "offset": -.5, "padding_width": 0, "width_rounding": "nearest_even", "min_content_width": 16}})
    base_config_path = DIRECTORY / "base-config.json"
    if not args.detected:
        base_config_path.write_text(json.dumps(base_config, ensure_ascii=False, indent=2) + "\n")
    records = []
    source_models = {}
    env = {**os.environ, "RAYON_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory(prefix="ocracle-style-eval-") as temporary:
        snapshot = Path(temporary)
        executable = snapshot / "ocracle"
        shutil.copy2(args.executable, executable)
        (snapshot / "models").mkdir()
        shutil.copy2(ROOT / "models/profiles.json", snapshot / "models/profiles.json")
        for key in (["en"] if args.detected else ["en", "en-mtg"]):
            profile = profiles[key]
            for field in ["model", "configuration"]:
                destination = snapshot / profile[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / profile[field], destination)
            source_models[key] = {"profile": key, "model_sha256": digest(snapshot / profile["model"]), "configuration_sha256": digest(snapshot / profile["configuration"]), "configuration": json.loads((snapshot / profile["configuration"]).read_text())}
        base = snapshot / "models/text-recognition-base.rten"
        if args.detected:
            detector = snapshot / "models/text-detection.onnx"
            shutil.copy2(ROOT / "models/text-detection.onnx", detector)
            source_models["en"]["detector_sha256"] = digest(detector)
        else:
            shutil.copy2(ROOT / "models/text-recognition-base.rten", base)
            shutil.copy2(base_config_path, snapshot / "base-config.json")
            source_models["ocrs-base"] = {"model_sha256": digest(base), "configuration_sha256": digest(base_config_path), "configuration": base_config}
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
                if key == "ocrs-base":
                    command += ["--recognition-model", str(base), "--model-config", str(snapshot / "base-config.json")]
                else:
                    command += ["--language", key, "--model-root", str(snapshot)]
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
    report = {"schema_version": 1, "evaluated_on": "2026-10-19", "status": "frozen_challenge_no_training_or_profile_readiness_claim", "inference_mode": "detected_text" if args.detected else "forced_single_line", "manifest_sha256": digest(DIRECTORY / "manifest.jsonl"), "provenance_sha256": digest(DIRECTORY / "provenance.json"), "executable_sha256": executable_sha, "integrity_verifier_stdout": integrity.stdout, "models": source_models, "preprocessing": "Actual CLI --passes all; original and enhanced candidates plus the engine's confidence selection retained. Expected text used only after inference for scoring.", "metrics": "NFC plus collapsed whitespace, preserving case/punctuation; CER can exceed100%. Casefold exact is a separately labelled diagnostic, never primary exact.", "confidence": "Uncalibrated geometric mean of emitted CTC posterior peaks; no correctness probability. Higher confidence need not be more accurate.", "limitations": "All complete title fields are kept in their original curved/warped layout. Rectangular crops retain annotated neighbouring text/mana/artwork. Failure mixes glyph, segmentation and layout limitations. No upper bound on future models' capability; upstream training membership unknown.", "summary": summary, "records": records}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
