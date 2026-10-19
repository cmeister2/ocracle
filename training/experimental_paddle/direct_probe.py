#!/usr/bin/env python3
"""Test the original color model in the real Rust engine on 58 frozen crops."""

from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

from probe import decode, digest, distance, normalized

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def samples():
    """Read existing fixtures without changing their sources, pixels or truth."""
    fixture_path = ROOT / "fixtures.json"
    fixtures = json.loads(fixture_path.read_text())
    rows = [{**x, "image": ROOT / x["path"], "language": x["lang"], "image_sha256": x["sha256"], "cohort": "cjk"} for x in fixtures["cases"]]
    latin = REPO / "training/latin-eval"
    for line in (latin / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows.append({**row, "image": latin / row["path"], "id": f"latin-{row['language']}-{row['card_id']}-{row['severity']}", "cohort": "latin"})
    folder = REPO / "docs/gallery/assets"
    gallery = json.loads((folder / "fixtures.json").read_text())
    for sample in gallery["samples"]:
        for variant in sample["variants"]:
            image = folder / variant["image"]
            rows.append({"id": sample["id"] + "-" + variant["id"], "image": image, "language": sample["language"], "image_sha256": digest(image), "card_id": sample["source"]["print_id"], "severity": "severe" if variant["id"] == "blur" else "clean", "text": sample["ground_truth"], "typography": sample["font"], "cohort": "gallery"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cli", type=Path, default=REPO / "target/release/ocracle")
    parser.add_argument("--output", type=Path, default=ROOT / "results-v6-direct.json")
    args = parser.parse_args()
    cache = ROOT / "cache"
    model = cache / "inference.onnx"
    assert digest(model) == "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"
    executable = cache / "ocracle-direct-probe-snapshot"
    shutil.copy2(args.cli, executable)
    config = json.loads((ROOT / "model-config.json").read_text())
    config.update({"model_name": "Paddle PP-OCRv6 small (upstream color, experimental MTG evaluation)", "output_layout": "NTC", "output_values": "probabilities", "input": {"color_mode": "bgr", "scale": 2 / 255, "offset": -1, "padding_width": 320, "width_rounding": "ceil"}})
    config_path = ROOT / "direct-model-config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    records = []
    env = {**os.environ, "RAYON_NUM_THREADS": "2"}
    for sample in samples():
        image = sample["image"]
        assert digest(image) == sample["image_sha256"]
        bgr = cv2.imread(str(image))
        h, w = bgr.shape[:2]
        rw = math.ceil(48 * w / h)
        width = max(320, rw)
        x = np.zeros((1, 3, 48, width), dtype=np.float32)
        x[0, :, :, :rw] = (cv2.resize(bgr, (rw, 48)).transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5
        output = session.run(None, {"x": x})[0]
        input_path, output_path = cache / "direct-input.f32", cache / "direct-output.f32"
        x.tofile(input_path)
        process = subprocess.run([str(args.runner.resolve()), str(model), str(input_path), str(width), str(output_path)], check=True, text=True, capture_output=True, env=env)
        rten_output = np.fromfile(output_path, dtype="<f4").reshape(output.shape)
        assert np.isfinite(rten_output).all()
        case_config = cache / "direct-case-config.json"
        case_config.write_text(json.dumps({**config, "language": sample["language"]}, ensure_ascii=False))
        cli_run = subprocess.run([str(executable), "recognize", str(image), "--recognition-model", str(model), "--model-config", str(case_config), "--line", "--passes", "original"], check=True, text=True, capture_output=True, env=env)
        cli = json.loads(cli_run.stdout)
        assert cli["model"]["sha256"] == digest(model)
        assert cli["model"]["language"] == sample["language"]
        expected = normalized(sample["text"])
        methods = {"official_bgr_ort": decode(output, config["alphabet"]), "official_bgr_rten": decode(rten_output, config["alphabet"]), "actual_cli_bgr_rten": {"text": cli["text"], "confidence": cli["confidence"], "method": cli["method"]}}
        for result in methods.values():
            result.update({"exact": normalized(result["text"]) == expected, "character_edits": distance(expected, normalized(result["text"]))})
        records.append({"id": sample["id"], "cohort": sample["cohort"], "language": sample["language"], "card_id": sample["card_id"], "severity": sample["severity"], "typography": sample.get("typography"), "expected": sample["text"], "expected_characters": len(expected), "image_sha256": sample["image_sha256"], "image_path": str(image.relative_to(REPO)), "input_shape": list(x.shape), "output_shape": list(output.shape), "ort_rten_identical_argmax": bool(np.array_equal(output.argmax(-1), rten_output.argmax(-1))), "ort_rten_max_abs_probability_error": float(np.abs(output - rten_output).max()), "results": methods})
        print(sample["id"], {k: v["text"] for k, v in methods.items()}, flush=True)
    summary = {}
    for method in records[0]["results"]:
        summary[method] = {}
        groups = {"all": records}
        for field in ["cohort", "severity", "language"]:
            for value in sorted({r[field] for r in records}):
                groups[value] = [r for r in records if r[field] == value]
        for group, rows in groups.items():
            edits = sum(r["results"][method]["character_edits"] for r in rows)
            characters = sum(r["expected_characters"] for r in rows)
            summary[method][group] = {"cases": len(rows), "exact": sum(r["results"][method]["exact"] for r in rows), "character_edits": edits, "characters": characters, "cer": edits / characters}
    report = {"schema_version": 1, "verified_on": "2026-10-19", "status": "experimental_not_comprehensive", "original_model_sha256": digest(model), "configuration_template_sha256": digest(config_path), "cli_sha256": digest(executable), "rten_version": "0.26.0", "packages": {p: importlib.metadata.version(p) for p in ["onnxruntime", "numpy", "Pillow", "opencv-python-headless"]}, "comparison": "NFC plus collapsed whitespace; preserve accents, case, punctuation and fullwidth digits. Original pass only, no fitting or confidence selection.", "preprocessing_difference": "Official ORT and isolated RTen share exact OpenCV INTER_LINEAR BGR tensors. Actual CLI uses image crate triangle resize; width ceiling, BGR order, normalized zero padding and affine normalization match model contract.", "identity_limitation": "Canonical Opt identity appears in prior custom Latin V1 training; Japanese crops are transfer/runtime probes, not globally canonical-unseen evidence. Upstream Paddle pretraining prints are unknown. No current model is fine-tuned by this evaluation.", "fixture_manifests": {str(p.relative_to(REPO)): digest(p) for p in [ROOT / "fixtures.json", REPO / "training/latin-eval/manifest.jsonl", REPO / "docs/gallery/assets/fixtures.json"]}, "summary": summary, "records": records}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
