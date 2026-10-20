# Isolated inference runtime diagnostic

These harnesses compare RTen and ONNX Runtime CPU inference on byte-identical
synthetic tensors. They measure model execution through the ordinary runtime API,
including its output creation, and exclude input creation, file reads, numerical
validation, resizing, detector postprocessing and CTC. They do not measure OCR
accuracy, phone performance or application frame rates.

Build the Rust executable once, then run the two runtimes sequentially with no
other active benchmark processes:

```sh
cargo build --release --locked --example inference_latency

python3 scripts/inference_latency.py \
  --model models/text-detection.onnx --shape 1,1,800,600 \
  --rten-executable target/release/examples/inference_latency \
  --output-dir /tmp/ocracle-runtime-detector

python3 scripts/inference_latency.py \
  --model models/ppocr-v6-small.onnx --shape 1,3,48,320 --classes 18710 \
  --rten-executable target/release/examples/inference_latency \
  --output-dir /tmp/ocracle-runtime-small
```

Use a Python environment with the NumPy/ONNX Runtime dependencies pinned in
`training/requirements.txt`. Repeat the recognition command with
`models/ppocr-v6-medium.onnx` to compare the larger recognizer. The default is four
intra-op threads, one cold call, three further warmups, and 20 measured calls.
ORT uses its CPU provider, sequential graph execution, one inter-op thread, and
all ordinary graph optimizations. RTen uses its normal model loader and an
explicit `RTEN_NUM_THREADS` value. The driver runs the RTen subprocess to completion
before constructing the ORT session, avoiding competing worker pools.

Inputs follow `float32((index % 251) / 128 - 1)` in contiguous row-major order.
Every value is exactly representable in float32. Both model and input hashes must
match. The output includes raw samples, upper-middle median, nearest-rank p95,
load/cold costs, shape, min/max/mean, first values, output hashes and complete
little-endian float32 dumps. Warm outputs must remain identical within each
runtime. `--classes` additionally checks the final class dimension and categorical
probability range/sums. Cross-runtime comparison checks every output value and
reports absolute differences, tolerance-based equality and argmax disagreements
(or the count of detector elements crossing the Ocrs threshold of 0.2).

Any speed difference is an inference diagnostic on synthetic data. Python and
Rust API overhead differ; optimized runtime kernels can also introduce small
numerical differences. A production backend change would still need real-image
parity, complete OCR latency/accuracy, integration/deployment work and sustained
Android measurements. No production runtime dependency changes with this probe.
