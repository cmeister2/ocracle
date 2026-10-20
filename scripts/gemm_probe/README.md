# Isolated RTen kernel comparison

RTen 0.26.0 exposes no production environment variable or public API to disable
its SIMD GEMM dispatch. `RTEN_BENCH_FILTER` applies only to upstream's internal
benchmark tests. The generic fallback is auto-vectorization-friendly and is
**not a scalar-only implementation**.

`prepare.py` checks the cached `rten-gemm-0.26.0.crate` against its registry
SHA-256 from this repository's Cargo.lock, checks its upstream commit, and checks
every extracted source file against that archive. It creates a separate project
containing a copy of the crate with just three declarations made public:
`F32KernelType`, `Int8KernelType`, and `WithKernel`. Project dependencies, the
Cargo registry cache and runtime kernel algorithms remain unchanged.

Prepare and build (requires the normal Cargo dependencies already fetched):

```sh
python3 scripts/gemm_probe/prepare.py \
  --crate-source ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rten-gemm-0.26.0 \
  --output /tmp/ocracle-gemm-probe
cargo build --release --locked --offline \
  --manifest-path /tmp/ocracle-gemm-probe/Cargo.toml
```

Use the corresponding registry-cache directory if Cargo's source identifier
differs on your machine. Preparation refuses an existing destination and copies
the experiment's pinned Cargo.lock. Source and probe digests are recorded in
`/tmp/ocracle-gemm-probe/provenance.json`.

Run when no OCR benchmarks or builds are active:

```sh
RAYON_NUM_THREADS=1 /tmp/ocracle-gemm-probe/target/release/ocracle-gemm-probe 20 \
  > docs/performance/gemm-kernels.jsonl
```

The default kernel and generic kernel process identical float32 matrices. Shapes
resemble a pointwise convolution and a recognition classifier; they do not
represent a full neural operator or OCR invocation. Packing is timed, input and
output allocation is not, each case warms three times, and raw samples are
retained. All selected-kernel outputs must equal the generic outputs for these
exactly representable inputs. Sixteen dot products also undergo independent
validation.

This shows the effect of the pinned runtime's optimized kernels on the current
host. It does not establish a pure SIMD instruction speedup, ARM NEON speedup,
end-to-end OCR acceleration or Android FPS.
