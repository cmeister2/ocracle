#!/usr/bin/env python3
"""Run or verify small/medium all-pass inference on the frozen fresh blur cohort."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time
import unicodedata

if not __debug__:
    raise RuntimeError("Evidence verification requires Python without -O")

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
CORPUS = ROOT / "training/blur-corpus"
RESULTS = DIRECTORY / "results-blur-development.json"
CACHE = DIRECTORY / "cache/blur-evaluation"
MANIFEST_SHA256 = "cea621376a6e6a7667196011837d6c96d61f8299f0ee8fc82b86c4a698aa47b6"
ANNOTATIONS_SHA256 = "34ff43f45a7d4587c6b5f278e5d6383e3de057db43d69f9f1d6f7319c40da737"
PROVENANCE_SHA256 = "ab5b0b73ce847a181ec733d2b11764817f7d1661f7e0dfe3cd2fb4f2bbea0c81"
CLI_SHA256 = "1ede5e7d7a8c2060114ab471adf040d965d0e697eb55f8d8eb094a66b6b28e13"
MODEL_IDS = ("v6-small", "v6-medium")
MODES = ("original", "enhanced", "selected")
DECISION_PROTOCOL = {
    "recorded_before_result_inspection": True,
    "timing_note": "Root supplied thresholds after evaluation launch, before any predictions, metrics or progress log were inspected.",
    "overall_selected_cer": "medium must strictly decrease across all448 observations",
    "degraded_selected_cer": "medium must strictly decrease across384 non-clean observations",
    "max_clean_cer_increase": .005,
    "max_each_language_cer_increase": .01,
    "exact_match_role": "secondary",
    "download_tradeoff": "76,554,979 bytes vs 21,159,378 bytes acceptable if all gates pass",
    "eligible_languages": ["de", "en", "es", "fr", "it", "pt"],
    "excluded_languages": ["la", "ja", "zhs", "zht"],
    "frozen_tests_used": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return digest_bytes(path.read_bytes())


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def normalized(text):
    """NFC and whitespace only: case, accents and punctuation remain literal."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for index, character in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[-1] + 1, previous[column - 1] + (character != other)))
        previous = current
    return previous[-1]


def inputs():
    for name, expected in [("manifest.jsonl", MANIFEST_SHA256), ("annotations.json", ANNOTATIONS_SHA256), ("provenance.json", PROVENANCE_SHA256)]:
        require(digest(CORPUS / name) == expected, f"Frozen corpus changed: {name}")
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").read_text().splitlines()]
    require(len(manifest) == 448, "Expected all 448 frozen observations")
    for case in manifest:
        require(digest(CORPUS / case["path"]) == case["image_sha256"], f"Changed fixture: {case['path']}")
    return manifest


def model_definitions():
    sources = json.loads((DIRECTORY / "sources.json").read_text())
    medium = next(row for row in sources["candidates"] if row["id"] == "v6-medium")
    small = sources["baseline"]
    definitions = {
        "v6-small": {"path": ROOT / "models/ppocr-v6-small.onnx", "sha256": small["model_sha256"], "bytes": small["model_bytes"], "repository": small["repository"], "revision": small["revision"]},
        "v6-medium": {"path": DIRECTORY / medium["files"]["inference.onnx"]["path"], "sha256": medium["files"]["inference.onnx"]["sha256"], "bytes": medium["files"]["inference.onnx"]["bytes"], "repository": medium["repository"], "revision": medium["revision"]},
    }
    for model_id, definition in definitions.items():
        path = DIRECTORY / "configs" / f"{model_id}.json"
        definition["template"] = json.loads(path.read_text())
        definition["template_sha256"] = digest(path)
    return definitions


def finite_confidences(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "confidence":
                require(isinstance(child, (int, float)) and not isinstance(child, bool) and math.isfinite(child) and 0 <= child <= 1, "Invalid confidence")
            finite_confidences(child)
    elif isinstance(value, list):
        for child in value:
            finite_confidences(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Non-finite inference value")


def selected_candidate(inference):
    candidates = inference["candidates"]
    require(len(candidates) == 2, "All-pass evaluation must retain both candidates")
    require(candidates[0]["method"].startswith("original + "), "First candidate is not original")
    require(candidates[1]["method"].startswith("2x Lanczos + contrast stretch + unsharp mask + "), "Second candidate is not the production enhancement")
    # Rust max_by chooses the last maximum on an equal posterior.
    index, selected = max(enumerate(candidates), key=lambda item: (item[1]["confidence"], item[0]))
    for key in ["text", "confidence", "method", "lines"]:
        require(inference[key] == selected[key], f"Selected {key} differs from highest-confidence candidate")
    return index


def metrics(records):
    grouped = defaultdict(lambda: {"observations": 0, "exact": 0, "edits": 0, "characters": 0})
    for record in records:
        reference = normalized(record["reference"])
        candidates = record["inference"]["candidates"]
        predictions = [candidates[0]["text"], candidates[1]["text"], record["inference"]["text"]]
        groups = ["all", f"condition/{record['condition']}", f"language/{record['language']}", f"style/{record['style']}", "italic/yes" if record["italic"] else "italic/no"]
        for mode, prediction in zip(MODES, predictions, strict=True):
            prediction = normalized(prediction)
            for group in groups:
                counts = grouped[f"{record['model']}/{mode}/{group}"]
                counts["observations"] += 1
                counts["characters"] += len(reference)
                counts["edits"] += distance(reference, prediction)
                counts["exact"] += reference == prediction
    return {key: {**value, "cer": value["edits"] / value["characters"]} for key, value in sorted(grouped.items())}


def timings(records):
    result = {}
    for model_id in MODEL_IDS:
        seconds = sorted(record["wall_seconds"] for record in records if record["model"] == model_id)
        result[model_id] = {"processes": len(seconds), "min_seconds": seconds[0], "median_seconds": statistics.median(seconds), "p95_seconds": seconds[math.ceil(.95 * len(seconds)) - 1], "max_seconds": seconds[-1], "sum_process_wall_seconds": sum(seconds)}
    return result


def tradeoffs(records):
    result = {}
    for model_id in MODEL_IDS:
        rows = [row for row in records if row["model"] == model_id]
        original = [normalized(row["reference"]) == normalized(row["inference"]["candidates"][0]["text"]) for row in rows]
        selected = [normalized(row["reference"]) == normalized(row["inference"]["text"]) for row in rows]
        result[model_id] = {"enhanced_selected": sum(row["selected_pass"] == "enhanced" for row in rows), "exact_helped_vs_original": sum(not a and b for a, b in zip(original, selected, strict=True)), "exact_harmed_vs_original": sum(a and not b for a, b in zip(original, selected, strict=True))}
    return result


def promotion_assessment(summary):
    """Apply the root's fixed, pre-result CER decision rule without retuning."""
    def group(name):
        values = {model: summary[f"{model}/selected/{name}"] for model in MODEL_IDS}
        return {"observations_per_model": values["v6-small"]["observations"], "small_cer": values["v6-small"]["cer"], "medium_cer": values["v6-medium"]["cer"], "medium_minus_small_cer": values["v6-medium"]["cer"] - values["v6-small"]["cer"]}

    overall, clean = group("all"), group("condition/clean")
    degraded_values = {}
    for model in MODEL_IDS:
        overall_counts = summary[f"{model}/selected/all"]
        clean_counts = summary[f"{model}/selected/condition/clean"]
        edits = overall_counts["edits"] - clean_counts["edits"]
        characters = overall_counts["characters"] - clean_counts["characters"]
        degraded_values[model] = {"edits": edits, "characters": characters, "cer": edits / characters}
    degraded = {"observations_per_model": 384, "small_cer": degraded_values["v6-small"]["cer"], "medium_cer": degraded_values["v6-medium"]["cer"], "medium_minus_small_cer": degraded_values["v6-medium"]["cer"] - degraded_values["v6-small"]["cer"]}
    languages = {language: group(f"language/{language}") for language in DECISION_PROTOCOL["eligible_languages"]}
    gates = {
        "overall_cer_strictly_decreases": overall["medium_cer"] < overall["small_cer"],
        "degraded_cer_strictly_decreases": degraded["medium_cer"] < degraded["small_cer"],
        "clean_cer_increase_at_most_0p5_percentage_points": clean["medium_minus_small_cer"] <= DECISION_PROTOCOL["max_clean_cer_increase"],
        "every_measured_language_cer_increase_at_most_1_percentage_point": all(row["medium_minus_small_cer"] <= DECISION_PROTOCOL["max_each_language_cer_increase"] for row in languages.values()),
    }
    return {"overall": overall, "degraded": degraded, "clean": clean, "languages": languages, "gates": gates, "medium_eligible_for_measured_latin_aliases": all(gates.values())}


def check():
    manifest, definitions = inputs(), model_definitions()
    report = json.loads(RESULTS.read_text())
    require(report["manifest_sha256"] == MANIFEST_SHA256 and report["annotations_sha256"] == ANNOTATIONS_SHA256 and report["provenance_sha256"] == PROVENANCE_SHA256, "Wrong frozen cohort")
    require(report["cli_sha256"] == CLI_SHA256, "Wrong immutable CLI snapshot")
    require(report["jobs"] == 2 and report["rayon_num_threads"] == 2, "Changed process contract")
    require(report["passes"] == "all" and report["line_mode"] is True, "Changed inference contract")
    require(report["decision_protocol"] == DECISION_PROTOCOL, "Changed pre-result decision rule")
    require(set(report["models"]) == set(MODEL_IDS), "Wrong predefined comparison models")
    cases = {(case["base_id"], case["condition"]): case for case in manifest}
    expected = {(model, *case) for model in MODEL_IDS for case in cases}
    languages = {case["language"] for case in manifest}
    for model_id, definition in definitions.items():
        model = report["models"][model_id]
        require(model["model_sha256"] == definition["sha256"] and model["model_bytes"] == definition["bytes"], "Changed official checkpoint pin")
        require(model["configuration_template_sha256"] == definition["template_sha256"], "Changed tensor/dictionary template")
        require(model["repository"] == definition["repository"] and model["revision"] == definition["revision"], "Changed official source attribution")
        require(set(model["configurations"]) == languages, "Wrong language configuration coverage")
        for language in languages:
            config = encode_json({**definition["template"], "language": language})
            require(model["configurations"][language]["sha256"] == digest_bytes(config), "Wrong per-language configuration")
        if definition["path"].exists():
            require(digest(definition["path"]) == definition["sha256"], "Local model changed")
    seen = set()
    for record in report["records"]:
        key = record["model"], record["base_id"], record["condition"]
        require(key not in seen and key in expected, "Unexpected/duplicate model-case record")
        seen.add(key)
        case = cases[record["base_id"], record["condition"]]
        for field in ["base_id", "condition", "language", "card_id", "oracle_id", "card_name", "style", "italic", "image_sha256", "source_image_sha256", "source_pixel_sha256", "clean_pixel_sha256", "bbox", "transform", "path"]:
            require(record[field] == case[field], f"Fixture identity or transformation changed: {field}")
        require(record["reference"] == case["text"], "Changed literal reference")
        require(record["returncode"] == 0 and isinstance(record["stderr"], str), "Inference process failed")
        require(record["stdout_sha256"] == digest_bytes(record["stdout"].encode()), "Changed raw stdout")
        require(record["stderr_sha256"] == digest_bytes(record["stderr"].encode()), "Changed raw stderr")
        require(json.loads(record["stdout"]) == record["inference"], "Parsed inference differs from raw stdout")
        inference = record["inference"]
        finite_confidences(inference)
        require(inference["confidence_kind"] == "uncalibrated geometric mean of emitted CTC character posterior peaks", "Changed confidence definition")
        index = selected_candidate(inference)
        require(record["selected_pass"] == MODES[index], "Wrong selected-pass label")
        require(inference["image_size"] == case["image_size"], "Wrong input dimensions")
        definition = definitions[record["model"]]
        require(inference["model"]["sha256"] == definition["sha256"] and record["model_sha256"] == definition["sha256"], "Wrong inference checkpoint")
        require(inference["model"]["language"] == case["language"], "Wrong inference language")
        require(inference["model"]["name"] == definition["template"]["model_name"], "Wrong inference configuration name")
        suffix = definition["template"]["architecture"] + " / greedy CTC"
        require(inference["candidates"][0]["method"] == "original + " + suffix, "Wrong original architecture/method")
        require(inference["candidates"][1]["method"] == "2x Lanczos + contrast stretch + unsharp mask + " + suffix, "Wrong enhanced architecture/method")
        require(record["configuration_sha256"] == report["models"][record["model"]]["configurations"][case["language"]]["sha256"], "Wrong configuration digest")
        require(record["cli_sha256"] == CLI_SHA256, "Wrong per-process executable")
        command = record["command"]
        require(len(command) == 10 and command[1] == "recognize" and command[3] == "--recognition-model" and command[5] == "--model-config" and command[7:] == ["--line", "--passes", "all"], "Wrong recorded inference arguments")
        require(command[2].endswith("/training/blur-corpus/" + case["path"]), "Wrong recorded image argument")
        require(Path(command[4]).name == record["model"] + ".onnx", "Wrong recorded checkpoint argument")
        require(Path(command[6]).name == record["model"] + "-" + case["language"] + ".json", "Wrong recorded configuration argument")
        require(math.isfinite(record["wall_seconds"]) and record["wall_seconds"] > 0, "Invalid process timing")
    require(seen == expected, "Missing model/case observations")
    require(report["summary"] == metrics(report["records"]), "Incorrect exact/CER summary")
    require(report["timings"] == timings(report["records"]), "Incorrect timing summary")
    require(report["selection_tradeoffs"] == tradeoffs(report["records"]), "Incorrect selection tradeoffs")
    require(report["promotion_assessment"] == promotion_assessment(report["summary"]), "Incorrect fixed-rule promotion assessment")
    snapshot = CACHE / "ocracle"
    if snapshot.exists():
        require(digest(snapshot) == CLI_SHA256, "Local CLI snapshot changed")
    print("Verified 2 official models × 448 fixed cases, both raw passes, selected posterior maximum and all metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path("/tmp/ocracle-before-colour-same-compiler"))
    parser.add_argument("--check", action="store_true", help="Offline validation; no inference or downloads")
    args = parser.parse_args()
    if args.check:
        check()
        return
    manifest, definitions = inputs(), model_definitions()
    require(digest(args.cli) == CLI_SHA256, "CLI differs from the predeclared engine snapshot")
    CACHE.mkdir(parents=True, exist_ok=True)
    executable = CACHE / "ocracle"
    shutil.copy2(args.cli, executable)
    models = {}
    languages = sorted({case["language"] for case in manifest})
    for model_id, definition in definitions.items():
        source = definition["path"]
        require(digest(source) == definition["sha256"], "Official model bytes differ from frozen pin")
        snapshot = CACHE / f"{model_id}.onnx"
        shutil.copy2(source, snapshot)
        models[model_id] = {"model_sha256": definition["sha256"], "model_bytes": source.stat().st_size, "repository": definition["repository"], "revision": definition["revision"], "configuration_template_sha256": definition["template_sha256"], "configurations": {}}
        for language in languages:
            config = CACHE / f"{model_id}-{language}.json"
            config.write_bytes(encode_json({**definition["template"], "language": language}))
            models[model_id]["configurations"][language] = {"sha256": digest(config), "derived_from": f"configs/{model_id}.json; only language set to {language}"}
    tasks = [(model_id, case) for case in manifest for model_id in MODEL_IDS]

    def infer(task):
        model_id, case = task
        config = CACHE / f"{model_id}-{case['language']}.json"
        model = CACHE / f"{model_id}.onnx"
        command = [str(executable), "recognize", str(CORPUS / case["path"]), "--recognition-model", str(model), "--model-config", str(config), "--line", "--passes", "all"]
        start = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True, timeout=120, env={**os.environ, "RAYON_NUM_THREADS": "2"})
        elapsed = time.monotonic() - start
        # Persist each raw process before parsing, including failure output.
        raw = {"command": command, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "wall_seconds": elapsed}
        (CACHE / f"raw-{model_id}-{case['base_id']}-{case['condition']}.json").write_bytes(encode_json(raw))
        require(process.returncode == 0, f"Inference failed: {model_id} {case['path']}: {process.stderr}")
        inference = json.loads(process.stdout)
        finite_confidences(inference)
        index = selected_candidate(inference)
        return {
            "model": model_id,
            **{field: case[field] for field in ["base_id", "condition", "language", "card_id", "oracle_id", "card_name", "style", "italic", "image_sha256", "source_image_sha256", "source_pixel_sha256", "clean_pixel_sha256", "bbox", "transform", "path"]},
            "reference": case["text"], "model_sha256": definition_for(model_id)["sha256"], "configuration_sha256": digest(config), "cli_sha256": CLI_SHA256,
            **raw, "stdout_sha256": digest_bytes(process.stdout.encode()), "stderr_sha256": digest_bytes(process.stderr.encode()), "selected_pass": MODES[index], "inference": inference,
        }

    def definition_for(model_id):
        return definitions[model_id]

    started = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, record in enumerate(pool.map(infer, tasks), 1):
            records.append(record)
            if index % 32 == 0:
                print(f"Completed {index}/{len(tasks)} processes in {time.monotonic() - started:.1f}s", flush=True)
    inputs()
    require(digest(executable) == CLI_SHA256, "CLI changed during evaluation")
    for model_id, definition in definitions.items():
        require(digest(CACHE / f"{model_id}.onnx") == definition["sha256"], "Model snapshot changed during evaluation")
    report = {
        "schema_version": 1, "evaluated_on": "2026-10-19", "partition": "development",
        "manifest": "training/blur-corpus/manifest.jsonl", "manifest_sha256": MANIFEST_SHA256, "annotations_sha256": ANNOTATIONS_SHA256, "provenance_sha256": PROVENANCE_SHA256,
        "cli_sha256": CLI_SHA256, "cli_source": "Immutable pre-colour-experiment production engine snapshot; original BGR input and current grayscale enhancement", "rten_version": "0.26.0",
        "jobs": 2, "rayon_num_threads": 2, "line_mode": True, "passes": "all", "models": models, "decision_protocol": DECISION_PROTOCOL,
        "metric_policy": "NFC and collapsed whitespace only; retain case, literal accents and punctuation. CER is total Levenshtein edits divided by total reference characters.",
        "selection_policy": "Uncalibrated geometric mean of emitted CTC character posterior peaks; maximum confidence across original/enhanced; last candidate wins equal-confidence tie. Not an accuracy probability.",
        "predeclaration": "All448 frozen development observations and both official pretrained checkpoints; no OCR-based source/label/condition filtering. Frozen gallery/accent/rare/poster tests not evaluated.",
        "timing_policy": "Wall time of separate CLI processes, including model load and both passes, with two concurrent processes and two Rayon threads each. These are process timings, not a throughput benchmark.",
        "host": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
        "evaluation_wall_seconds": time.monotonic() - started, "summary": metrics(records), "timings": timings(records), "selection_tradeoffs": tradeoffs(records), "records": records,
    }
    report["promotion_assessment"] = promotion_assessment(report["summary"])
    RESULTS.write_bytes(encode_json(report))
    check()


if __name__ == "__main__":
    main()
