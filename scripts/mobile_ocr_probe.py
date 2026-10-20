#!/usr/bin/env python3
"""Experimental Python DB + CTC OCR pipeline, not Android production code.

Harness interface: mobile_ocr_probe.py MODEL CONFIG IMAGE [ITERATIONS] [line|full]
Use --make-config INFERENCE_YML --config-output CONFIG_JSON --language en to
make a performance_suite.py-compatible sidecar without running inference.

Pre/postprocessing is adapted from PaddleX (Apache-2.0), pinned at
c852371c701d83cd21955b2963834825920ffebb. See mobile_ocr_probe.md and the
accompanying license. Model graphs and official inference.yml files are inputs;
the probe does not download models, modify graphs, or consult reference text.
"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Modifications Copyright (c) 2026 ocracle contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
import yaml

PADDLEX_REVISION = "c852371c701d83cd21955b2963834825920ffebb"
IMPLEMENTATION = "experimental_python_paddle_db_ctc"
SOURCE_ROOT = f"https://github.com/PaddlePaddle/PaddleX/tree/{PADDLEX_REVISION}"


def sha256(path: Path) -> str:
    """Hash the exact bytes used for an experiment."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    """Record identity and verify a downloaded published hash when available."""
    result = {"path": str(path.resolve()), "sha256": sha256(path)}
    provenance_path = path.parent / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        record = provenance.get("files", {}).get(path.name)
        if record:
            if result["sha256"] != record["sha256"]:
                raise ValueError(f"Artifact differs from recorded source: {path}")
            result.update({
                "repository": provenance.get("repository"),
                "revision": provenance.get("revision"),
                "source_url": record.get("url"),
                "published_lfs_sha256": record.get("published_lfs_sha256"),
            })
    return result


def official_config(path: Path) -> tuple[dict, dict]:
    """Read an official YAML/JSON config or a hash-pinned harness sidecar."""
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text())
        if "official_configuration" in config:
            reference = config["official_configuration"]
            source = Path(reference["path"])
            if not source.is_absolute():
                source = path.parent / source
            if sha256(source) != reference["sha256"]:
                raise ValueError(f"Official configuration hash differs: {source}")
            return official_config(source)
    else:
        config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not all(
        key in config for key in ("Global", "PreProcess", "PostProcess")
    ):
        raise ValueError("Expected an official Paddle inference configuration")
    return config, artifact(path)


def transform(config: dict, name: str, default=None):
    """Find a named transform without relying on YAML operation ordering."""
    for operation in config["PreProcess"]["transform_ops"]:
        if name in operation:
            return operation[name] or {}
    if default is not None:
        return default
    raise ValueError(f"Official configuration is missing {name}")


def validate_bgr(config: dict) -> None:
    """Reject unsupported pixel contracts instead of silently changing colors."""
    decode = transform(config, "DecodeImage")
    if decode.get("img_mode") != "BGR" or decode.get("channel_first", False):
        raise ValueError("This prototype supports official HWC BGR configurations")


def detector_input(image: np.ndarray, config: dict, max_side: int) -> np.ndarray:
    """Match PaddleX standalone max-side resize and BGR affine normalization."""
    height, width = image.shape[:2]
    if height + width < 64:
        padded = np.zeros((max(32, height), max(32, width), 3), dtype=np.uint8)
        padded[:height, :width] = image
        image = padded
        height, width = image.shape[:2]
    ratio = min(1., max_side / max(height, width))
    resized_height = max(round(int(height * ratio) / 32) * 32, 32)
    resized_width = max(round(int(width * ratio) / 32) * 32, 32)
    if (resized_height, resized_width) != (height, width):
        image = cv2.resize(image, (resized_width, resized_height))
    normalize = transform(config, "NormalizeImage")
    scale = normalize.get("scale", "1./255.")
    if scale in ("1./255.", "1/255", "1.0/255.0"):
        scale = 1. / 255.
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("Unsupported detector normalization scale")
    if normalize.get("order", "hwc") != "hwc":
        raise ValueError("Expected official HWC detector normalization")
    means = normalize.get("mean", [0.485, 0.456, 0.406])
    deviations = normalize.get("std", [0.229, 0.224, 0.225])
    channels = list(cv2.split(image))
    for index in range(3):
        channels[index] = channels[index].astype(np.float32)
        channels[index] *= scale / deviations[index]
        channels[index] += -means[index] / deviations[index]
    normalized = cv2.merge(channels)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    """Return PaddleX's ordered min-area quadrilateral and its shorter side."""
    rectangle = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(rectangle), key=lambda point: point[0])
    first, fourth = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    second, third = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return np.array([points[first], points[second], points[third], points[fourth]]), min(rectangle[1])


def box_score(probabilities: np.ndarray, box: np.ndarray) -> float:
    """Average original posterior pixels inside the minimum-area rectangle."""
    height, width = probabilities.shape
    box = box.copy()
    x0 = max(0, min(math.floor(box[:, 0].min()), width - 1))
    x1 = max(0, min(math.ceil(box[:, 0].max()), width - 1))
    y0 = max(0, min(math.floor(box[:, 1].min()), height - 1))
    y1 = max(0, min(math.ceil(box[:, 1].max()), height - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    box[:, 0] -= x0
    box[:, 1] -= y0
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(probabilities[y0:y1 + 1, x0:x1 + 1], mask)[0]


def db_boxes(probabilities: np.ndarray, original_size: tuple[int, int], config: dict) -> list[tuple[np.ndarray, float]]:
    """Reproduce configured PaddleX DB quadrilateral decoding and unclipping."""
    post = config["PostProcess"]
    if post.get("name") != "DBPostProcess":
        raise ValueError("Expected a DB detector")
    if post.get("score_mode", "fast") != "fast" or post.get("box_type", "quad") != "quad":
        raise ValueError("This probe supports fast-score quadrilateral DB output")
    segmentation = probabilities > post.get("thresh", 0.3)
    if post.get("use_dilation", False):
        segmentation = cv2.dilate(
            segmentation.astype(np.uint8), np.array([[1, 1], [1, 1]])
        )
    contours, _ = cv2.findContours(
        (segmentation * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    original_height, original_width = original_size
    mask_height, mask_width = probabilities.shape
    boxes = []
    for contour in contours[:post.get("max_candidates", 1000)]:
        points, short_side = mini_box(contour)
        if short_side < 3:
            continue
        score = box_score(probabilities, points)
        if score < post.get("box_thresh", 0.6):
            continue
        distance = cv2.contourArea(points) * post.get("unclip_ratio", 2.) / cv2.arcLength(points, True)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        paths = offset.Execute(distance)
        if not paths:
            continue
        try:
            expanded = np.array(paths)
        except ValueError:
            expanded = np.array(paths[0])
        box, short_side = mini_box(expanded.reshape(-1, 1, 2).astype(np.float32))
        if short_side < 5:
            continue
        for point in box:
            point[0] = max(0, min(round(point[0] * (original_width / mask_width)), original_width))
            point[1] = max(0, min(round(point[1] * (original_height / mask_height)), original_height))
        boxes.append((box.astype(np.int16), float(score)))
    return boxes


def sort_boxes(boxes: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
    """Match PaddleX top-to-bottom sorting with its ten-pixel row adjustment."""
    boxes = sorted(boxes, key=lambda item: (item[0][0][1], item[0][0][0]))
    for index in range(len(boxes) - 1):
        for previous in range(index, -1, -1):
            left, right = boxes[previous][0], boxes[previous + 1][0]
            if abs(int(right[0][1]) - int(left[0][1])) < 10 and right[0][0] < left[0][0]:
                boxes[previous], boxes[previous + 1] = boxes[previous + 1], boxes[previous]
            else:
                break
    return boxes


def crop_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Use the official min-area refit, cubic perspective crop and tall rotation."""
    points, _ = mini_box(np.asarray(quad).astype(np.int32))
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    if width <= 0 or height <= 0:
        raise ValueError("Detector produced a degenerate recognition crop")
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    mapping = cv2.getPerspectiveTransform(points, destination)
    cropped = cv2.warpPerspective(
        image, mapping, (width, height), borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if cropped.shape[0] / cropped.shape[1] >= 1.5:
        cropped = np.rot90(cropped)
    return cropped


def recognition_input(image: np.ndarray, shape: list[int]) -> np.ndarray:
    """Match PaddleX dynamic recognition resizing and normalized zero padding."""
    channels, height, minimum_width = shape
    source_height, source_width = image.shape[:2]
    if channels != 3 or min(source_height, source_width, height, minimum_width) <= 0:
        raise ValueError("Invalid three-channel recognition image/shape")
    maximum_ratio = max(minimum_width / height, source_width / source_height)
    width = int(height * maximum_ratio)
    if width > 3200:
        width = resized_width = 3200
    else:
        ratio = source_width / float(source_height)
        resized_width = min(width, math.ceil(height * ratio))
    resized = cv2.resize(image, (resized_width, height)).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255
    resized -= 0.5
    resized /= 0.5
    padded = np.zeros((channels, height, width), np.float32)
    padded[:, :, :resized_width] = resized
    return padded


def ctc_decode(probabilities: np.ndarray, characters: list[str]) -> list[tuple[str, float]]:
    """Match Paddle CTC first-argmax ties and first-emission arithmetic scores."""
    if probabilities.ndim != 3 or probabilities.shape[2] != len(characters):
        raise ValueError("Recognition output shape does not match the official alphabet")
    labels = probabilities.argmax(axis=-1)
    scores = probabilities.max(axis=-1)
    decoded = []
    for row, confidence in zip(labels, scores, strict=True):
        selected = np.ones(len(row), dtype=bool)
        selected[1:] = row[1:] != row[:-1]
        selected &= row != 0
        text = "".join(characters[label] for label in row[selected])
        score = float(confidence[selected].mean()) if selected.any() else 0.
        if not math.isfinite(score):
            raise ValueError("Nonfinite recognition confidence")
        decoded.append((text, score))
    return decoded


def make_session(model: Path, threads: int) -> ort.InferenceSession:
    """Create one explicit CPU session, outside the warm pipeline measurements."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected a single-input, single-output exported model")
    return session


class MobileOcr:
    """Reusable experimental CPU sessions and official preprocessing contracts."""

    def __init__(self, model: Path, configuration: Path, detector: Path | None,
                 detector_configuration: Path | None, max_side: int, threads: int, batch_size: int):
        self.rec_config, self.rec_config_identity = official_config(configuration)
        validate_bgr(self.rec_config)
        post = self.rec_config["PostProcess"]
        if post.get("name") != "CTCLabelDecode" or not post.get("character_dict"):
            raise ValueError("Expected the official CTC character dictionary")
        self.characters = ["blank", *post["character_dict"], " "]
        self.rec_shape = transform(self.rec_config, "RecResizeImg")["image_shape"]
        self.model_identity = artifact(model)
        self.recognizer = make_session(model, threads)
        self.detector = None
        self.detector_identity = None
        self.det_config_identity = None
        self.det_config = None
        if detector is not None:
            self.det_config, self.det_config_identity = official_config(
                detector_configuration or detector.with_name("inference.yml")
            )
            validate_bgr(self.det_config)
            self.detector_identity = artifact(detector)
            self.detector = make_session(detector, threads)
        self.max_side = max_side
        self.batch_size = batch_size

    def read(self, image: np.ndarray, single_line: bool, layout: str) -> dict:
        """Perform fresh preprocessing, model calls, decoding and geometry."""
        stages = {}
        detector_shape = None
        start = time.perf_counter()
        height, width = image.shape[:2]
        if layout == "standard_card" and not single_line:
            image = image.copy()
            image[height * 12 // 100:height * 54 // 100,
                  width * 4 // 100:width * 96 // 100] = 220
        stages["layout_ms"] = (time.perf_counter() - start) * 1000
        if single_line:
            boxes = [(np.array([[0, 0], [width, 0], [width, height], [0, height]], np.int32), None)]
            crops = [image]
        else:
            if self.detector is None:
                raise ValueError("Full OCR requires --detector or OCRACLE_DETECTOR_MODEL")
            start = time.perf_counter()
            tensor = detector_input(image, self.det_config, self.max_side)
            detector_shape = list(tensor.shape)
            stages["detector_preprocess_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            output = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
            stages["detector_inference_ms"] = (time.perf_counter() - start) * 1000
            if output.ndim != 4 or output.shape[:2] != (1, 1):
                raise ValueError("Expected a [1,1,H,W] DB probability output")
            start = time.perf_counter()
            boxes = sort_boxes(db_boxes(output[0, 0], (height, width), self.det_config))
            crops = [crop_quad(image, box) for box, _ in boxes]
            stages["detector_postprocess_and_crops_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        order = sorted(range(len(crops)), key=lambda index: crops[index].shape[1] / crops[index].shape[0])
        decoded = [None] * len(crops)
        batches = []
        for begin in range(0, len(order), self.batch_size):
            indices = order[begin:begin + self.batch_size]
            inputs = [recognition_input(crops[index], self.rec_shape) for index in indices]
            maximum_width = max(item.shape[2] for item in inputs)
            tensor = np.stack([
                np.pad(item, ((0, 0), (0, 0), (0, maximum_width - item.shape[2])))
                for item in inputs
            ])
            output = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: tensor})[0]
            for index, result in zip(indices, ctc_decode(output, self.characters), strict=True):
                decoded[index] = result
            batches.append(list(tensor.shape))
        stages["recognition_preprocess_inference_decode_ms"] = (time.perf_counter() - start) * 1000
        lines = []
        for (quad, detector_score), (text, confidence) in zip(boxes, decoded, strict=True):
            left = max(0, int(quad[:, 0].min()))
            top = max(0, int(quad[:, 1].min()))
            right = min(width, int(quad[:, 0].max()))
            bottom = min(height, int(quad[:, 1].max()))
            lines.append({
                "text": text, "confidence": confidence,
                "bounds": [left, top, right - left, bottom - top],
                "quad": quad.tolist(), "detector_confidence": detector_score,
            })
        return {
            "text": "\n".join(line["text"] for line in lines),
            "lines": lines, "stages_ms": stages, "recognition_batches": batches,
            "detector_input_shape": detector_shape,
        }


def main() -> None:
    """Expose a compatible latency harness and an inference-free config writer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("configuration", type=Path, nargs="?")
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("iterations", type=int, nargs="?", default=20)
    parser.add_argument("mode", choices=["line", "full"], nargs="?", default="full")
    parser.add_argument("--make-config", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--detector", type=Path, default=os.environ.get("OCRACLE_DETECTOR_MODEL"))
    parser.add_argument("--detector-config", type=Path, default=os.environ.get("OCRACLE_DETECTOR_CONFIG"))
    parser.add_argument("--max-side", type=int, default=int(os.environ.get("OCRACLE_DB_MAX_SIDE", "960")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("RTEN_NUM_THREADS", "4")))
    parser.add_argument("--rec-batch-size", type=int, default=int(os.environ.get("OCRACLE_REC_BATCH_SIZE", "6")))
    parser.add_argument("--layout", choices=["text", "standard_card"], default=os.environ.get("OCRACLE_LAYOUT", "text"))
    args = parser.parse_args()
    if args.make_config:
        if not args.config_output:
            parser.error("--make-config requires --config-output")
        config, identity = official_config(args.make_config)
        sidecar = {
            "implementation": IMPLEMENTATION, "paddlex_revision": PADDLEX_REVISION,
            "model_name": config["Global"]["model_name"], "language": args.language,
            "languages": args.languages or [args.language],
            "official_configuration": identity,
        }
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
        return
    if not all((args.model, args.configuration, args.image)):
        parser.error("MODEL CONFIG IMAGE are required")
    if min(args.iterations, args.threads, args.rec_batch_size) <= 0 or not 32 <= args.max_side <= 4000:
        parser.error("iterations/threads/batch must be positive; max-side must be in 32..4000")
    if os.environ.get("OCRACLE_BACKEND", "onnxruntime") != "onnxruntime":
        parser.error("This prototype requires OCRACLE_BACKEND=onnxruntime")
    native = sorted((Path(ort.__file__).parent / "capi").glob("libonnxruntime.so.*"))
    declared_native = os.environ.get("ORT_DYLIB_PATH")
    if declared_native and native and sha256(Path(declared_native)) != sha256(native[0]):
        raise ValueError(
            "ORT_DYLIB_PATH differs from this Python wheel's native library; "
            "set it to the reported wheel library for performance_suite provenance"
        )
    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode image: {args.image}")
    height, width = image.shape[:2]
    if height * width > 24_000_000 or max(height, width) > 32767:
        raise ValueError("Image exceeds supported size for Paddle int16 DB coordinates")
    single_line = args.mode == "line"
    start = time.perf_counter()
    engine = MobileOcr(
        args.model, args.configuration, None if single_line else args.detector,
        args.detector_config, args.max_side, args.threads, args.rec_batch_size,
    )
    load_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    reference = engine.read(image, single_line, args.layout)
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(3):
        engine.read(image, single_line, args.layout)
    samples, stage_samples = [], []
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = engine.read(image, single_line, args.layout)
        samples.append((time.perf_counter() - start) * 1000)
        stage_samples.append(result["stages_ms"])
        if result["text"] != reference["text"] or result["lines"] != reference["lines"]:
            raise ValueError("Nondeterministic pipeline output")
    ordered = sorted(samples)
    report = {
        "implementation": IMPLEMENTATION,
        "measurement_scope": "warm fresh Python OCR pipeline; excludes model loading and image decoding; not Android production or handset FPS",
        "source_contract": SOURCE_ROOT,
        "backend": "onnxruntime", "layout": args.layout, "single_line": single_line,
        "model": {
            "name": engine.rec_config["Global"]["model_name"],
            "sha256": engine.model_identity["sha256"],
            "detector_sha256": engine.detector_identity["sha256"] if engine.detector_identity else None,
        },
        "recognition_artifact": engine.model_identity,
        "recognition_configuration": engine.rec_config_identity,
        "detector_artifact": engine.detector_identity,
        "detector_configuration": engine.det_config_identity,
        "image": str(args.image), "image_sha256": sha256(args.image), "image_size": [width, height],
        "threads": args.threads, "opencv_threads": 1, "recognition_batch_size": args.rec_batch_size,
        "detector_max_side": args.max_side, "detector_limit_type": "max",
        "runtime": {"onnxruntime": ort.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                    "pyclipper": pyclipper.__version__, "python": platform.python_version(),
                    "native_library": artifact(native[0]) if native else None},
        "arch": platform.machine(), "os": platform.system(),
        "load_ms": load_ms, "cold_ms": cold_ms, "samples_ms": samples,
        "median_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "stage_samples_ms": stage_samples,
        "text": reference["text"], "lines": reference["lines"],
        "recognition_batches": reference["recognition_batches"],
        "detector_input_shape": reference["detector_input_shape"],
        "confidence_method": "Paddle arithmetic mean of first emitted timestep posteriors; uncalibrated",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
