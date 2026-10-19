#!/usr/bin/env python3
"""Gate native training on exact-weight forward agreement with deployed ONNX."""
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import onnxruntime as ort
import paddle
from native import HERE, ROOT, load_native


def main():
    paddle.set_device("cpu")
    paddle.seed(20261019)
    model, mapping = load_native()
    onnx = ROOT / "models/ppocr-v6-small.onnx"
    model_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
    if model_sha != "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634":
        raise RuntimeError("Wrong deployed ONNX checkpoint")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(20261019)
    records = []
    start = time.time()
    with paddle.no_grad():
        for batch, width, kind in [(1, 320, "zero"), (1, 320, "random"),
                                   (1, 513, "random"), (2, 320, "random")]:
            image = np.zeros((batch, 3, 48, width), dtype=np.float32)
            if kind == "random":
                image = rng.uniform(-1, 1, image.shape).astype(np.float32)
            native = model(paddle.to_tensor(image)).numpy()
            official = session.run(None, {"x": image})[0]
            shape = list(official.shape)
            error = float(np.max(np.abs(native - official)))
            identical = bool(np.array_equal(native.argmax(-1), official.argmax(-1)))
            record = {"batch": batch, "width": width, "input": kind, "output_shape": shape,
                      "max_absolute_probability_error": error, "identical_argmax": identical}
            records.append(record)
            print(json.dumps(record), flush=True)
            if native.shape != official.shape or error > .00005 or not identical:
                raise RuntimeError("Native weights/forward graph do not match deployed ONNX")
    report = {"status": "exact_weight_forward_parity_passed", "paddle_version": paddle.__version__,
              "onnxruntime_version": ort.__version__, "python": platform.python_version(),
              "onnx_sha256": model_sha, "native_checkpoint": mapping,
              "checks": records, "elapsed_seconds": round(time.time() - start),
              "limitations": ["Synthetic tensor parity, not recognition accuracy or training improvement.",
                              "Original auxiliary NRTR branch omitted; CTC branch only."]}
    (HERE / "parity.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
