#!/usr/bin/env python3
"""Compare official BGR inference, grayscale adaptation, RTen, and the real CLI."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import unicodedata

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def decode(probabilities, alphabet, log=False):
    rows = probabilities[0] if not log else probabilities[:, 0]
    labels = rows.argmax(-1)
    previous = 0
    text = []
    scores = []
    for index, row in zip(labels, rows):
        if index and index != previous:
            text.append(alphabet[index - 1])
            scores.append(float(row[index]))
        previous = index
    return {"text": "".join(text), "mean_first_emission_posterior": float(np.mean(np.exp(scores) if log else scores)) if scores else 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=REPO / "target/release/ocracle")
    parser.add_argument("--scope", choices=["fixed", "gallery", "japanese"], default="fixed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "model-config.json").read_text())
    alphabet = config["alphabet"]
    cache = ROOT / "cache"
    # Snapshot the executable so concurrent repository builds cannot change this run.
    executable = cache / "ocracle-probe-snapshot"
    shutil.copy2(args.cli, executable)
    if args.scope == "fixed":
        fixture_manifest = ROOT / "fixtures.json"
        fixtures = json.loads(fixture_manifest.read_text())
        samples = [{**x, "image": ROOT / x["path"], "language": x["lang"], "image_sha256": x["sha256"]} for x in fixtures["cases"]]
        latin = REPO / "training/latin-eval"
        for line in (latin / "manifest.jsonl").read_text().splitlines():
            row = json.loads(line)
            samples.append({**row, "image": latin / row["path"], "id": f"latin-{row['language']}-{row['card_id']}-{row['severity']}"})
    else:
        folder = REPO / ("docs/gallery/assets" if args.scope == "gallery" else "training/experimental_easyocr/fixtures")
        fixture_manifest = folder / "fixtures.json"
        fixtures = json.loads(fixture_manifest.read_text())
        samples = []
        for sample in fixtures["samples"]:
            for variant in sample["variants"]:
                image = folder / variant["image"]
                samples.append({"id": sample["id"] + "-" + variant["id"], "image": image, "language": sample["language"], "image_sha256": digest(image), "card_id": sample["source"]["print_id"], "severity": "severe" if variant["id"] == "blur" else "clean", "text": sample["ground_truth"], "typography": sample.get("font", sample.get("treatment"))})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    original_path = cache / "inference.onnx"
    wrapper_path = cache / "paddle-v6-small-gray-logctc.onnx"
    original = ort.InferenceSession(str(original_path), sess_options=options, providers=["CPUExecutionProvider"])
    wrapped = ort.InferenceSession(str(wrapper_path), sess_options=options, providers=["CPUExecutionProvider"])
    records = []
    env = {**os.environ, "RAYON_NUM_THREADS": "2"}
    for sample in samples:
        image = sample["image"]
        assert digest(image) == sample["image_sha256"], image
        bgr = cv2.imread(str(image))
        height, width = bgr.shape[:2]
        resized_width = math.ceil(48 * width / height)
        padded_width = max(320, resized_width)
        direct_input = np.zeros((1, 3, 48, padded_width), dtype=np.float32)
        direct_input[0, :, :, :resized_width] = (cv2.resize(bgr, (resized_width, 48)).transpose(2, 0, 1).astype(np.float32) / 255 - .5) * 2
        # Controlled Rec709 luminance plus OpenCV geometry. The snapshotted CLI
        # also has its own luminance implementation, width rounding and resize.
        rgb = bgr[:, :, ::-1].astype(np.uint32)
        gray = ((rgb * np.array([2126, 7152, 722], dtype=np.uint32)).sum(-1) // 10000).astype(np.uint8)
        gray_input = (cv2.resize(gray, (resized_width, 48)).astype(np.float32) / 255 - .5)[None, None]
        direct_output = original.run(None, {"x": direct_input})[0]
        wrapped_output = wrapped.run(None, {"input": gray_input})[0]
        input_path = cache / "probe-input.f32"
        output_path = cache / "probe-output.f32"
        gray_input.tofile(input_path)
        runner = subprocess.run([str(args.runner.resolve()), str(wrapper_path), str(input_path), str(resized_width), str(output_path)], check=True, text=True, capture_output=True, env=env)
        rten_output = np.fromfile(output_path, dtype="<f4").reshape(wrapped_output.shape)
        if not np.isfinite(rten_output).all():
            raise ValueError("RTen produced nonfinite values")
        case_config = cache / "case-config.json"
        case_config.write_text(json.dumps({**config, "language": sample["language"]}, ensure_ascii=False))
        cli_run = subprocess.run([str(executable), "recognize", str(image), "--recognition-model", str(wrapper_path), "--model-config", str(case_config), "--line", "--passes", "original"], check=True, text=True, capture_output=True, env=env)
        cli = json.loads(cli_run.stdout)
        assert cli["model"]["sha256"] == digest(wrapper_path)
        expected = normalized(sample["text"])
        methods = {"official_bgr_ort": decode(direct_output, alphabet), "grayscale_wrapper_ort": decode(wrapped_output, alphabet, log=True), "grayscale_wrapper_rten": decode(rten_output, alphabet, log=True), "actual_cli_wrapper_rten": {"text": cli["text"], "confidence": cli["confidence"], "method": cli["method"]}}
        for result in methods.values():
            result["exact"] = normalized(result["text"]) == expected
            result["character_edits"] = distance(expected, normalized(result["text"]))
        record = {"id": sample["id"], "card_id": sample["card_id"], "language": sample["language"], "severity": sample["severity"], "typography": sample.get("typography"), "image_path": str(image.relative_to(REPO)), "image_sha256": sample["image_sha256"], "expected": sample["text"], "expected_characters": len(expected), "missing_alphabet": sorted(set(expected) - set(alphabet)), "controlled_input_shape": list(gray_input.shape), "output_shape": list(wrapped_output.shape), "ort_rten_max_abs_log_error": float(np.abs(wrapped_output - rten_output).max()), "ort_rten_identical_argmax": bool(np.array_equal(wrapped_output.argmax(-1), rten_output.argmax(-1))), "rten_runner": runner.stdout.strip(), "results": methods}
        records.append(record)
        print(sample["id"], {k: v["text"] for k, v in methods.items()}, flush=True)
    summary = {}
    for method in records[0]["results"]:
        by_group = {}
        groups = {"all": records}
        for severity in {r["severity"] for r in records}:
            groups[severity] = [r for r in records if r["severity"] == severity]
        for language in {r["language"] for r in records}:
            groups[language] = [r for r in records if r["language"] == language]
        for name, rows in groups.items():
            edits = sum(r["results"][method]["character_edits"] for r in rows)
            characters = sum(r["expected_characters"] for r in rows)
            by_group[name] = {"cases": len(rows), "exact": sum(r["results"][method]["exact"] for r in rows), "character_edits": edits, "characters": characters, "cer": edits / characters}
        summary[method] = by_group
    report = {"schema_version": 1, "verified_on": "2026-10-19", "scope": args.scope, "fixture_manifest_sha256": digest(fixture_manifest), "status": "experimental_not_registered_or_comprehensive", "original_model_sha256": digest(original_path), "wrapper_model_sha256": digest(wrapper_path), "config_template_sha256": digest(ROOT / "model-config.json"), "cli_sha256": digest(executable), "packages": {p: importlib.metadata.version(p) for p in ["onnx", "onnxruntime", "numpy", "Pillow", "opencv-python-headless", "PyYAML"]}, "comparison": "NFC plus collapsed whitespace; preserve accents, case and punctuation. No checkpoint fitting/selection. All original-pass results retained.", "confidence_note": "Probe posterior is arithmetic mean of first emission steps; CLI is geometric mean of peak posterior in each emitted run. Both uncalibrated; do not compare as accuracy.", "summary": summary, "records": records}
    output = args.output or ROOT / ("results-v6-small.json" if args.scope == "fixed" else f"results-v6-{args.scope}.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
