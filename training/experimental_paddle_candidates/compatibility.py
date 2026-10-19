#!/usr/bin/env python3
"""Check original ONNX graphs against RTen on two fixed development tensors and padding."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml

from evaluate import DIRECTORY, FIXTURES, candidates, digest, normalized


def decode(output, alphabet):
    previous = 0
    characters = []
    for label in output[0].argmax(-1):
        if label and label != previous:
            characters.append(alphabet[label - 1])
        previous = label
    return "".join(characters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(FIXTURES.read_text())
    # This sampling rule depends only on the frozen labels/styles, never OCR output.
    title = next(s for s in manifest["samples"] if s["kind"] == "title")
    italic = next(s for s in manifest["samples"] if s["italic"])
    fixtures = [(title, "clean"), (italic, "downsample")]
    tensors = []
    for sample, variant_id in fixtures:
        variant = next(v for v in sample["variants"] if v["id"] == variant_id)
        path = FIXTURES.parent / variant["path"]
        if digest(path) != variant["sha256"]:
            raise ValueError("Changed development crop")
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        content_width = math.ceil(48 * width / height)
        tensor = np.zeros((1, 3, 48, max(320, content_width)), dtype=np.float32)
        tensor[:, :, :, :content_width] = (
            cv2.resize(image, (content_width, 48)).transpose(2, 0, 1).astype(np.float32) / 255 - .5) * 2
        tensors.append((sample["id"] + "-" + variant_id, tensor, sample["text"], variant["sha256"]))
    tensors.append(("normalized-zero-padding-width320", np.zeros((1, 3, 48, 320), dtype=np.float32), None, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    records = []
    for model in candidates():
        if model["id"] != "v6-small":
            dictionary = yaml.safe_load((DIRECTORY / "sources" / model["id"] / "inference.yml").read_text())["PostProcess"]["character_dict"]
            if model["config"]["alphabet"] != "".join(dictionary) + " ":
                raise ValueError("Candidate dictionary order changed")
        graph = onnx.load(model["model"])
        session = ort.InferenceSession(str(model["model"]), sess_options=options, providers=["CPUExecutionProvider"])
        for case_id, tensor, reference, image_sha in tensors:
            expected = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            input_path, output_path = DIRECTORY / "cache/parity-input.f32", DIRECTORY / "cache/parity-output.f32"
            tensor.tofile(input_path)
            run = subprocess.run([str(args.runner.resolve()), str(model["model"]), str(input_path), str(tensor.shape[-1]), str(output_path)],
                                 check=True, capture_output=True, text=True, env={**os.environ, "RAYON_NUM_THREADS": "2"})
            actual = np.fromfile(output_path, dtype="<f4").reshape(expected.shape)
            if not np.isfinite(actual).all() or expected.shape[-1] != len(model["config"]["alphabet"]) + 1:
                raise ValueError("Invalid CTC output")
            if not np.array_equal(expected.argmax(-1), actual.argmax(-1)):
                raise ValueError("ORT and RTen greedy paths disagree")
            records.append({"model": model["id"], "model_sha256": digest(model["model"]), "case_id": case_id,
                            "image_sha256": image_sha, "reference": reference,
                            "input_shape": list(tensor.shape), "output_shape": list(actual.shape),
                            "opset": [{"domain": x.domain, "version": x.version} for x in graph.opset_import],
                            "max_absolute_probability_difference": float(np.max(np.abs(expected - actual))),
                            "identical_argmax": True, "rten_row_sum_max_error": float(np.max(np.abs(actual.sum(-1) - 1))),
                            "official_bgr_ort_text": decode(expected, model["config"]["alphabet"]),
                            "official_bgr_rten_text": decode(actual, model["config"]["alphabet"]),
                            "runner_stdout": run.stdout, "runner_stderr": run.stderr})
            print(model["id"], case_id, records[-1]["official_bgr_rten_text"], flush=True)
    report = {"schema_version": 1, "verified_on": "2026-10-19", "rten_version": "0.26.0",
              "runner_sha256": digest(args.runner), "fixture_manifest_sha256": digest(FIXTURES),
              "selection": "First title/clean, first italic/downsample in frozen development manifest, plus a normalized-zero width320 tensor; selected without OCR. The zero tensor is a runtime probe, not an accuracy case.",
              "normalization": "Official BGR OpenCV INTER_LINEAR -> NCHW float32 ((pixel/255)-.5)*2; aspect ceil width, minimum zero-padded320. Identical input tensors for both runtimes.",
              "production_difference": "Full development accuracy report uses actual Rust CLI Triangle resizing; these shared-tensor parity checks use official OpenCV resizing.",
              "packages": {p: importlib.metadata.version(p) for p in ["onnx", "onnxruntime", "numpy", "opencv-python-headless", "PyYAML"]},
              "records": records}
    (DIRECTORY / "compatibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
