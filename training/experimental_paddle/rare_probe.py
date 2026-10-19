#!/usr/bin/env python3
"""Probe pinned official v5 and primary-author Hebrew models on reviewed scans."""

from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import re
import subprocess

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from probe import decode, digest, distance, normalized

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def arabic_logical(text):
    """Apply Paddle's documented Arabic group reversal without reordering Latin runs.

    Adapted from Apache-2.0 PaddleOCR BaseRecLabelDecode.pred_reverse; original
    authors PaddlePaddle, LICENSE.paddle. Kept separate from raw output in reports.
    """
    groups = []
    latin = ""
    for character in text:
        if re.fullmatch(r"[a-zA-Z0-9 :*./%+\-]", character):
            latin += character
        else:
            if latin:
                groups.append(latin)
                latin = ""
            groups.append(character)
    if latin:
        groups.append(latin)
    return "".join(reversed(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--only-language", choices=["he", "ar", "sa", "grc", "la"])
    parser.add_argument("--hebrew-model", choices=["mobile", "server"], default="mobile")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = REPO / "training/rare-scripts/manifest.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.only_language:
        samples = [sample for sample in samples if sample["language"] == args.only_language]
    inventory = json.loads((ROOT / "additional-models.json").read_text())
    keys = {"grc": "el", "ar": "arabic", "sa": "devanagari", "he": "hebrew", "la": "v6"}
    sessions = {}
    records = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    for sample in samples:
        key = keys[sample["language"]]
        folder = ROOT / "cache" / (key if key != "v6" else "")
        if key not in sessions:
            if key != "v6":
                specification = next(x for x in inventory["models"] if x["key"] == key)
                for file in specification["files"]:
                    assert digest(ROOT / "cache" / file["cache_path"]) == file["sha256"]
            if key == "hebrew":
                model = folder / ("mobile-word-rec.onnx" if args.hebrew_model == "mobile" else "server-svtrv2-ctc.onnx")
                alphabet = "".join((folder / "charset_v2f.txt").read_text().splitlines()) + " "
            else:
                model = folder / "inference.onnx"
                dictionary = yaml.safe_load((folder / "inference.yml").read_text())["PostProcess"]["character_dict"]
                alphabet = "".join(dictionary) + " "
            session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
            sessions[key] = (model, alphabet, session)
        model, alphabet, session = sessions[key]
        image = manifest.parent / sample["path"]
        assert digest(image) == sample["image_sha256"]
        bgr = cv2.imread(str(image))
        h, w = bgr.shape[:2]
        if key == "hebrew":
            rw = max(1, round(48 * w / h))
            width = min(3200, max(32, math.ceil(rw / 16) * 16))
            rw = min(rw, width)
            pixels = np.full((48, width, 3), 255, dtype=np.uint8)
            pixels[:, :rw] = cv2.resize(bgr, (rw, 48))
            x = ((pixels.transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5)[None]
        else:
            rw = math.ceil(48 * w / h)
            width = max(320, rw)
            x = np.zeros((1, 3, 48, width), dtype=np.float32)
            x[0, :, :, :rw] = (cv2.resize(bgr, (rw, 48)).transpose(2, 0, 1).astype(np.float32) / 255 - .5) / .5
        probabilities = session.run(None, {session.get_inputs()[0].name: x})[0]
        assert probabilities.shape[-1] == len(alphabet) + 1
        assert np.isfinite(probabilities).all()
        raw = decode(probabilities, alphabet)
        logical = arabic_logical(raw["text"]) if key == "arabic" else raw["text"]
        input_path = folder / "rare-input.f32"
        output_path = folder / "rare-output.f32"
        x.tofile(input_path)
        process = subprocess.run([str(args.runner.resolve()), str(model), str(input_path), str(width), str(output_path)], text=True, capture_output=True, env={**os.environ, "RAYON_NUM_THREADS": "2"})
        parity = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        if process.returncode == 0:
            rten = np.fromfile(output_path, dtype="<f4").reshape(probabilities.shape)
            parity.update({"finite": bool(np.isfinite(rten).all()), "max_abs_probability_error": float(np.abs(probabilities - rten).max()), "identical_argmax": bool(np.array_equal(probabilities.argmax(-1), rten.argmax(-1))), "raw_text": decode(rten, alphabet)["text"]})
        expected = normalized(sample["text"])
        record = {**sample, "probe_model": key, "model_sha256": digest(model), "input_shape": list(x.shape), "output_shape": list(probabilities.shape), "dictionary_characters_including_space": len(alphabet), "dictionary_duplicate_entries": len(alphabet) - len(set(alphabet)), "missing_expected_glyphs": sorted(set(expected) - set(alphabet)), "raw_ort": raw, "logical_text": logical, "logical_postprocess": "Paddle Arabic group reversal; raw output preserved" if key == "arabic" else "none; logical Unicode claimed by source", "exact": normalized(logical) == expected, "character_edits": distance(expected, normalized(logical)), "reference_characters": len(expected), "rten": parity}
        records.append(record)
        print(sample["id"], sample["severity"], repr(logical), 'expected', repr(sample["text"]), 'missing',record['missing_expected_glyphs'],'rten',process.returncode,flush=True)
    summary = {}
    for language in keys:
        rows = [x for x in records if x["language"] == language]
        if not rows:
            continue
        summary[language] = {"cases": len(rows), "exact": sum(x["exact"] for x in rows), "character_edits": sum(x["character_edits"] for x in rows), "reference_characters": sum(x["reference_characters"] for x in rows), "rten_successful_cases": sum(x["rten"]["returncode"] == 0 for x in rows)}
    report = {"schema_version": 1, "verified_on": "2026-10-19", "status": "experimental_not_registered_or_comprehensive", "hebrew_model": args.hebrew_model, "fixture_manifest_sha256": digest(manifest), "model_inventory_sha256": digest(ROOT / "additional-models.json"), "rten_version": "0.26.0", "packages": {p: importlib.metadata.version(p) for p in ["onnxruntime", "numpy", "opencv-python-headless", "PyYAML"]}, "scoring": "NFC plus collapsed whitespace; no mark stripping, no ASCII folding or speculative ligature replacement.", "summary": summary, "records": records}
    output = args.output or ROOT / "results-rare.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
