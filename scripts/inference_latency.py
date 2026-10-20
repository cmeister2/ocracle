#!/usr/bin/env python3
"""Compare ONNX Runtime and optional RTen on an identical synthetic input tensor.

This is an inference-only diagnostic, not OCR accuracy or application FPS.
Requires NumPy and ONNX Runtime (already pinned by training/requirements.txt).
Run comparisons sequentially while no other benchmarks are active.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import onnxruntime as ort


def sha256(data: bytes) -> str:
    """Hash model or little-endian tensor bytes."""
    return hashlib.sha256(data).hexdigest()


def output_summary(values: np.ndarray, classes: int | None, raw_path: Path) -> dict:
    """Inspect complete numerical output outside all timed inference calls."""
    if values.dtype != np.float32 or not values.size or not np.isfinite(values).all():
        raise ValueError("Output must be a nonempty, finite float32 tensor")
    categorical = None
    if classes is not None:
        if classes < 2 or values.shape[-1] != classes:
            raise ValueError("Output class axis differs from --classes")
        if np.min(values) < 0 or np.max(values) > 1.000001:
            raise ValueError("Output values are not probabilities")
        error = float(np.max(np.abs(values.astype(np.float64).sum(-1) - 1)))
        if error > 0.001:
            raise ValueError("Categorical probabilities are not normalized")
        categorical = {"classes": classes, "row_sum_max_error": error}
    raw = values.astype("<f4", copy=False).tobytes()
    raw_path.write_bytes(raw)
    return {
        "shape": list(values.shape), "elements": int(values.size),
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
        "first_values": values.ravel()[:8].tolist(), "sha256": sha256(raw),
        "categorical_probabilities": categorical, "identical_between_calls": True,
        "raw_f32le": str(raw_path),
    }


def measure_ort(args: argparse.Namespace, tensor: np.ndarray) -> tuple[dict, np.ndarray]:
    """Measure CPUExecutionProvider with explicit threads and normal graph optimization."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    model_bytes = args.model.read_bytes()
    start = time.perf_counter_ns()
    session = ort.InferenceSession(model_bytes, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter_ns() - start) / 1e6
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Diagnostic requires one input and one output")
    feed = {session.get_inputs()[0].name: tensor}
    start = time.perf_counter_ns()
    reference = session.run(None, feed)[0]
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    for _ in range(3):
        session.run(None, feed)
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        observed = session.run(None, feed)[0]
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if not np.array_equal(reference, observed):
            raise ValueError("Output values changed between identical inputs")
    ordered = sorted(samples)
    result = {
        "scope": "synthetic tensor inference only; not OCR/application or Android FPS",
        "runtime": "onnxruntime", "runtime_version": ort.__version__,
        "provider": "CPUExecutionProvider", "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL", "graph_optimization_level": "ORT_ENABLE_ALL",
        "model": str(args.model), "model_sha256": sha256(model_bytes),
        "input_shape": list(tensor.shape),
        "input_sha256": sha256(tensor.astype("<f4", copy=False).tobytes()),
        "input_formula": "float32((index % 251) / 128 - 1), contiguous row-major",
        "threads": args.threads, "arch": platform.machine(), "os": platform.system().lower(),
        "warmup_runs_after_cold": 3, "load_ms": load_ms, "cold_ms": cold_ms,
        "load_scope": "model parsing/optimization; file read excluded",
        "samples_ms": samples, "median_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "output": output_summary(reference, args.classes, args.output_dir / "ort-output.f32"),
    }
    return result, reference


def main() -> None:
    """Run RTen before constructing the ORT session, avoiding competing worker pools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shape", required=True, help="For example 1,1,800,600 or 1,3,48,320")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--classes", type=int, help="Validate final categorical-probability axis")
    parser.add_argument("--rten-executable", type=Path,
                        help="Optional compiled examples/inference_latency binary")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if not shape or min(shape) < 1 or math.prod(shape) > 100_000_000:
        parser.error("Shape must contain positive dimensions with at most 100M elements")
    if args.threads < 1 or args.iterations < 1:
        parser.error("Thread and iteration counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor = ((np.arange(math.prod(shape), dtype=np.int64) % 251).astype(np.float32)
              / np.float32(128) - np.float32(1)).reshape(shape)
    result = {
        "scope": "synthetic inference runtime diagnostic, not an application benchmark",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "order": "RTen subprocess first, then ONNX Runtime CPU session; no overlapping sessions",
        "limitations": "Synthetic input only; no resizing, OCR detection postprocessing, CTC, "
                      "camera, handset or thermal measurements. Python and Rust API overhead "
                      "differ. Runtime migration still requires real-image correctness and "
                      "end-to-end Android measurements.",
        "records": [],
    }
    rten_result = None
    if args.rten_executable:
        command = [str(args.rten_executable.resolve()), str(args.model.resolve()),
                   "--shape", args.shape, "--iterations", str(args.iterations),
                   "--output", str((args.output_dir / "rten-output.f32").resolve())]
        if args.classes is not None:
            command.extend(["--classes", str(args.classes)])
        environment = dict(os.environ)
        for name in ("RTEN_TIMING", "RTEN_TRACE"):
            environment.pop(name, None)
        environment["RTEN_NUM_THREADS"] = str(args.threads)
        completed = subprocess.run(command, env=environment, capture_output=True,
                                   text=True, check=True)
        rten_result = json.loads(completed.stdout)
        result["records"].append(rten_result)
        result["rten_executable_sha256"] = sha256(args.rten_executable.read_bytes())
    ort_result, ort_output = measure_ort(args, tensor)
    result["records"].append(ort_result)
    if rten_result:
        for key in ("model_sha256", "input_sha256", "input_shape", "threads"):
            if rten_result[key] != ort_result[key]:
                raise ValueError(f"Runtime comparison mismatch: {key}")
        if rten_result["output"]["shape"] != ort_result["output"]["shape"]:
            raise ValueError("Runtime output shapes differ")
        rten_output = np.fromfile(args.output_dir / "rten-output.f32", dtype="<f4")
        rten_output = rten_output.reshape(ort_output.shape)
        difference = np.abs(rten_output.astype(np.float64) - ort_output)
        comparison = {
            "rten_median_ms": rten_result["median_ms"], "ort_median_ms": ort_result["median_ms"],
            "rten_over_ort_median": rten_result["median_ms"] / ort_result["median_ms"],
            "bitwise_identical_outputs": rten_result["output"]["sha256"] == ort_result["output"]["sha256"],
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "allclose_atol_1e-5_rtol_1e-3": bool(np.allclose(
                rten_output, ort_output, atol=1e-5, rtol=1e-3)),
        }
        if args.classes:
            mismatch = np.argmax(rten_output, axis=-1) != np.argmax(ort_output, axis=-1)
            comparison["argmax_steps"] = int(mismatch.size)
            comparison["argmax_steps_different"] = int(mismatch.sum())
        else:
            comparison["threshold_0_2_elements_different"] = int(
                np.count_nonzero((rten_output > 0.2) != (ort_output > 0.2)))
        result["comparison"] = comparison
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
