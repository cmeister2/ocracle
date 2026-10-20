# Android optimization decisions

Investigated 2026-10-20 against the pinned models and RTen 0.26.0. The
[baseline](README.md) measures a desktop; papers and runtime capabilities do not
establish 15 FPS on an unspecified phone. The target is 66.7 ms per **fresh**
OCR result, including detection, recognition, preprocessing and decoding.

## SIMD and model execution

The installed crates' `.cargo_vcs_info.json` identifies RTen commit
`14cbf4620476f402237bb29f731a244d3b6bed4d`. Inspection confirms float32 NEON
GEMM, int8 NEON fallback, and runtime dispatch to `dotprod`/`i8mm` kernels on
supported AArch64 processors. SIMD is already active in these inference kernels;
there is no missing application “enable NEON” switch. Preserve runtime dispatch
instead of globally requiring optional ISA features on unknown handsets.
[Pinned GEMM dispatch](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/lib.rs),
[AArch64 kernels](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/rten-gemm/src/kernels/aarch64.rs).

The first experiments should reduce total neural work: smaller detector input,
less recognition padding, and cheaper class decoding. Hand-written SIMD in
preprocessing is lower priority until a stage profile shows its contribution.
Compare 1/2/4 inference threads on-device; desktop thread scaling cannot predict
thermal behavior or scheduling across unlike phone cores. RTen supports both
thread-count control and operator timing.
[RTen performance guidance](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/docs/performance.md).

There is no production kernel override environment variable in 0.26.0. The
private kernel selector can be exposed in an isolated, verified crate copy for
a [reproducible generic-versus-selected GEMM comparison](../../scripts/gemm_probe/README.md).
This generic fallback can be auto-vectorized; it is not a scalar-only baseline.
On this desktop with one Rayon thread, the selected AVX512 kernel took 0.184 ms
versus generic 0.731 ms for the synthetic pointwise shape, and 7.329 versus
21.284 ms for the recognition-head shape. All computed outputs matched; these
packing-inclusive synthetic results support retaining the optimized dispatch.
[Raw samples](gemm-kernels.jsonl), [probe provenance](gemm-provenance.json).

## Quantization feasibility and reproducible candidates

RTen supports `ConvInteger`, `MatMulInteger` and dynamic quantization, but not
`QLinearConv`/`QLinearMatMul`. Its preferred integer inputs are uint8 activations
and int8 weights. Dynamic conversion adds activation scanning and conversion;
dot-product support alone does not guarantee a speedup, particularly for small
or depthwise operators. Range reduction is relevant to x86 without VNNI; do not
infer Arm quantized accuracy from a saturating x86 run.
[Pinned quantization guide](https://github.com/robertknight/rten/blob/14cbf4620476f402237bb29f731a244d3b6bed4d/docs/quantization.md).

Local ONNX introspection found:

| Original artifact | Bytes | Conv | ConvTranspose | MatMul | Public input |
| --- | ---: | ---: | ---: | ---: | --- |
| Detector | 2,499,479 | 53 | 6 | 0 | `[1,1,800,600]` |
| PP-OCRv6 small | 21,159,378 | 57 | 0 | 13 | `[N,3,48,W]` |
| PP-OCRv6 medium | 76,554,979 | 62 | 0 | 13 | `[N,3,48,W]` |

All convolution weights and nine of each recognizer's MatMul right operands
are constant. Both recognizers emit 18,710 classes. The detector's fixed input
metadata means simply shrinking the caller's image does not establish smaller
inference dimensions.

[`scripts/quantize_ocr.py`](../../scripts/quantize_ocr.py) requires the source
digest and new output/report paths, refuses overwrites, checks ONNX structure
and preserves the public tensor contract. It changes no registry or defaults.
With `onnx==1.23.2` and `onnxruntime==1.31.0`, dynamic signed int8 export produced:

| Candidate | Bytes | ConvInteger | MatMulInteger | Evidence |
| --- | ---: | ---: | ---: | --- |
| Detector | 1,896,177 | 53 | 0 | [Identity/graph report](quantization-detector.json) |
| Small | 5,613,990 | 57 | 9 | [Identity/graph report](quantization-small.json) |
| Medium | 19,541,116 | 62 | 9 | [Identity/graph report](quantization-medium.json) |

Although per-channel quantization was requested, this exporter produces scalar
Conv weight zero points and vector MatMul weight zero points. Six detector
transposed convolutions and four recognizer attention MatMuls remain float.
These are structurally valid experiments, not accuracy-qualified models.

```sh
# Use a Python environment with training/requirements.txt's ONNX dependencies.
python scripts/quantize_ocr.py models/ppocr-v6-small.onnx \
  /tmp/ocracle-small-dynamic-int8.onnx \
  --source-sha256 5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634 \
  --report /tmp/ocracle-small-dynamic-int8.json
```

Use distinct output paths for `--matmul-only`, `--reduce-range`, or
`--uint8-weights` comparisons. Test text error and full-card localization before
adopting a candidate. Static calibration or quantization-aware training could
improve the tradeoff, but requires representative camera data and a compatible
export format. ORT recommends preprocessing and distinguishes dynamic from
calibration-based static quantization.
[ORT quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## Papers and other runtimes

| Direction | Evidence and actionable decision |
| --- | --- |
| Lightweight text detector | [Differentiable Binarization](https://arxiv.org/abs/1911.08947) studies efficient segmentation and simpler postprocessing. Evaluate input-size/recall tradeoffs and a mobile detector candidate; its reported benchmark FPS is not Android evidence. |
| Efficient recognizer | [PP-OCRv3](https://arxiv.org/abs/2206.03001) combines SVTR-LCNet with training improvements. [SVTR](https://arxiv.org/abs/2205.00159) is a recognizer architecture comparison, not a drop-in runtime optimization. Smaller models need the repository's Latin/blur accuracy gate. |
| Improve small-model accuracy | [PP-OCRv5](https://arxiv.org/abs/2603.24373) argues that data quality/diversity can make a compact OCR system competitive. Explore domain distillation/fine-tuning if the small recognizer fails the accuracy floor; this paper does not describe the pinned v6 artifacts. |
| Alternative CPU backend | [ORT mobile](https://onnxruntime.ai/docs/tutorials/mobile/) recommends CPU for quantized models and XNNPACK for float models as initial choices. Benchmark the same ONNX graphs, shapes, thread count and end-to-end boundaries before accepting extra JNI/runtime complexity. |
| GPU | [LiteRT GPU documentation](https://developers.google.com/edge/litert/performance/gpu) warns that unsupported operators split execution and synchronization can erase gains. [ncnn](https://github.com/Tencent/ncnn) offers mobile CPU/Vulkan execution. Conversion parity, operator coverage, upload/readback and sustained device timing are necessary. |
| NNAPI/NPU | [Android's migration guide](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) confirms NNAPI deprecation beginning with Android 15 and discusses alternatives. Avoid making a new NNAPI integration the default; evaluate current accelerators against actual supported devices. |

## Camera and device validation

Use CameraX `STRATEGY_KEEP_ONLY_LATEST`, one in-flight OCR request and a reusable
engine. Release each `ImageProxy` after consuming its pixels. This bounds
backlog while OCR is busy; it does not turn a slow OCR pass into 15 freshly
recognized frames per second.
[CameraX image analysis](https://developer.android.com/media/camera/camerax/analyze).

Tracking regions or reusing unchanged text may make a preview responsive, but
report fresh OCR rate separately from overlay refresh rate. Validate invalidation
on card replacement, camera motion, focus and exposure changes. Preserve color
for this BGR-trained recognizer; measure conversion from camera planes rather
than assuming the grayscale detector's path can replace recognition input.

The official Linux NDK r30 archive was downloaded and verified against Google's
published size/checksum. The release library and latency harness then linked
successfully for ARM64/API24, with vector instructions present in the executable.
[Android build evidence and reproduction](android-build.md) records toolchain,
source and artifact identities. No handset, APK integration or device execution
was available; linking does not establish runtime behavior or frame rate.

On a chosen handset, record SoC/OS/ABI, app and model identities, thread count,
camera resolution/rotation, cold start, warm p50/p95, detection and recognition
times, dropped frames and camera-to-result age. Repeat under sustained load with
temperature/power context. The acceptance workload must include full cards,
blur, perspective and small printed text; tiny title crops are insufficient.
