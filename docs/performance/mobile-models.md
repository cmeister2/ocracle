# Reproduce lightweight mobile OCR candidates

The immediate target is approximately **15 title OCR updates/second**. Test the
recognizers on title crops with the existing Rust `--line` path first. A title
crop benchmark excludes finding that title in a camera frame; report that
boundary explicitly. Whole-card OCR and the DB detector pilot are supplementary
experiments.

The [pinned manifest](mobile-models.json) records five original ONNX exports
from PaddlePaddle's official Hugging Face organization. It includes complete
revision IDs, immutable URLs, file sizes, SHA-256 digests, publisher-declared
Apache-2.0 licenses, graph contracts and expected generated configuration hashes.
ONNX digests matched the publisher's LFS metadata. Original graphs are unchanged;
these are pretrained candidates, without MTG fine-tuning or registry changes.

| Candidate | ONNX size | Output classes | Float tensor constants |
| --- | ---: | ---: | ---: |
| [V5 English recognizer](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx/tree/3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b) | 7.85 MB | 438 | 1,941,617 |
| [V5 Latin recognizer](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/tree/89d3a50e2c27e2e7cceeab0e944c25c807d5db4f) | 8.04 MB | 838 | 1,990,017 |
| [V6 tiny recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/tree/2612ab37152ae0a677521bae4e1e3d4fb4cf7c30) | 4.46 MB | 6,906 | 1,104,524 |
| [V5 mobile DB detector](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/tree/e6f4fa85f00e168c862bc462aebca69eef9b3d3d) | 4.83 MB | 1 mask | 1,171,745 |
| [V6 tiny DB detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/tree/2ba1506c0380b8f0b03dd142459aac66d4421f6c) | 1.78 MB | 1 mask | 428,420 |

Float counts include both initializers and Constant-node tensors; they are not
strictly trainable-parameter counts. All five graphs passed ONNX 1.23.2 validation
and have no external tensor data. These static checks establish neither runtime
compatibility nor recognition quality or speed.

## Fetch and reproduce the configurations

Use a separate Python environment; only PyYAML is needed by this script:

```sh
python3 -m venv /tmp/ocracle-model-fetch
/tmp/ocracle-model-fetch/bin/python -m pip install PyYAML==6.0.3
/tmp/ocracle-model-fetch/bin/python scripts/fetch_mobile_models.py
```

The default destination is `.cache/performance/mobile-models/`. Each model
directory contains the exact publisher `inference.onnx`, `inference.yml` and
`README.md`; recognizers additionally get `ocracle-config.json`.
`ocracle-candidate.json` records the pinned candidate, manifest identity and
generated-file hashes. The script verifies every original file before parsing
configuration data and verifies generated recognition configuration hashes.
Existing files must match; changed or corrupted files are not overwritten.

Select candidates or reuse an existing download directory:

```sh
/tmp/ocracle-model-fetch/bin/python scripts/fetch_mobile_models.py \
  --model PP-OCRv6_tiny_rec_onnx \
  --model latin_PP-OCRv5_mobile_rec_onnx \
  --output-dir /tmp/ocracle-mobile-candidates

# No network access; still reproduce/verify configurations from cached YAML.
/tmp/ocracle-model-fetch/bin/python scripts/fetch_mobile_models.py \
  --model PP-OCRv6_tiny_rec_onnx \
  --model latin_PP-OCRv5_mobile_rec_onnx \
  --output-dir /tmp/ocracle-mobile-candidates --offline
```

Use a fresh output directory after changing the pinned manifest or experiment
configuration. The script never performs inference or changes default profiles.

## Run title recognition through the Rust engine

After building the release latency example, this example measures the ordinary
RTen path on a fixed title crop:

```sh
cargo build --release --locked --example latency
RTEN_NUM_THREADS=4 OCRACLE_BACKEND=rten \
  target/release/examples/latency \
  .cache/performance/mobile-models/PP-OCRv6_tiny_rec_onnx/inference.onnx \
  .cache/performance/mobile-models/PP-OCRv6_tiny_rec_onnx/ocracle-config.json \
  docs/gallery/assets/beleren-title.png 20 line
```

Select the other candidate directories to hold the fixture and timing boundary
constant. For optional ONNX Runtime CPU execution, build with
`--features onnxruntime`, set `OCRACLE_BACKEND=onnxruntime`, and supply an absolute
`ORT_DYLIB_PATH` to the target-platform library. The same
`scripts/performance_suite.py` can run these models/configurations over
`--workload line`; retain model, config, executable and runtime identities.
Sweep one, two and four threads, check title text on clean and degraded fixtures,
and run comparisons sequentially.

The [measured title comparison](README.md#title-result-and-app-integration)
retains the resulting speed/accuracy tradeoffs and every prediction. Tiny is
much faster on the measured host but loses title accuracy; it has not replaced
the default model.

The Python probe using official Paddle preprocessing returned the same 13/16
exact tiny-model title results as Rust, including the same three failures. Its
different interpolation/width rounding did not resolve those failures on this
cohort. This was a separate five-call screening run; see
[`suite-python-tiny-title.json`](suite-python-tiny-title.json).

A measured crop median below 66.7 ms would support only that workload and host.
Sustained Android title-update performance still requires device measurements,
plus any title localization, frame conversion and scheduling used by the
application. The [Android measurement procedure](android-benchmark.md) defaults
to a title crop and describes whole-card mode separately.

## Recognition contract and known differences

All three recognizers accept BGR float32 `[N,3,48,W]`, normalize channels with
`pixel * (2/255) - 1`, and return NTC softmax probabilities. The generated
configurations preserve the original dictionary order, append one space, and
reserve class zero for CTC blank. English has 436 dictionary entries; Latin has
836 entries but only 772 unique characters. **Do not deduplicate the Latin
dictionary**: repeated glyphs still occupy distinct trained output classes.
V6 tiny has 6,904 unique entries.

The generated configurations use the current engine's triangle resize, ceiling
width, minimum padded width 320 and normalized-zero right padding. Pinned
PaddleX uses OpenCV linear interpolation, truncates its dynamic width calculation
above 320 and caps width at 3200; the Rust engine rejects widths above 4096.
Thus configurations fit the engine but do not reproduce official input tensors
bit for bit. Confidence aggregation also differs.
[PaddleX recognition processors](https://github.com/PaddlePaddle/PaddleX/blob/7f45ff529b396c865ad661a1e4d99d431c5d61f3/paddlex/inference/models/text_recognition/processors.py),
[predictor configuration](https://github.com/PaddlePaddle/PaddleX/blob/7f45ff529b396c865ad661a1e4d99d431c5d61f3/paddlex/inference/models/text_recognition/predictor.py).

English lacks `– Œ œ ß`; Latin lacks `–`. V6 tiny includes the queried glyphs
`—–“”‘’ÆæŒœéàöß`. Those observations describe dictionary coverage, not accuracy.
Generated configurations label the English evaluation profile; other languages
require their own accuracy evaluation and explicitly recorded configurations.

## Supplementary DB detector contract

The downloaded YAML remains the authority for each detector's parameters.
Both use three-channel BGR input and normalize with
`(pixel / 255 - mean) / std`, where mean is `[0.485,0.456,0.406]` and standard
deviation `[0.229,0.224,0.225]`, applied in that BGR order.
They cannot be substituted into the current grayscale Ocrs detector adapter.

| Published parameter | V5 mobile | V6 tiny |
| --- | ---: | ---: |
| Bitmap threshold | 0.3 | 0.2 |
| Box score threshold | 0.6 | 0.4 |
| Maximum candidates | 1000 | 3000 |
| Unclip ratio | 1.5 | 1.4 |
| YAML resize operation | `resize_long: 960` | Empty `DetResizeForTest` |

DB postprocessing produces text-region/line quadrilaterals. It requires
probability scoring, polygon expansion, coordinate mapping and recognition
crops; Ocrs word-box expansion/grouping is a different contract. The Python
`scripts/mobile_ocr_probe.py` pilot consumes these original `inference.yml`
files directly. Record
its selected maximum-side resize explicitly, because module and full-pipeline
defaults can differ. Preserve v5/v6 threshold differences.

The supplementary Python experiment paired the v6 tiny detector and recognizer,
with recognition batches of up to six lines and the standard artwork mask:

| Detector maximum side | Whole-card median ms | Scored-region CER |
| --- | ---: | ---: |
| 960 | 103.5 | 1.92% |
| 640 | 76.8 | 2.87% |

These are geometric means across the same three full scans, five measured warm
calls each, four CPU threads on the desktop host. All pipeline work is timed;
CER still covers only the fixed title/type/body regions. The 640 result loses
accuracy, including Goudy title text. Raw predictions, stage times, batch shapes
and dependency/model identities are retained in
[`suite-python-tiny-card960.json`](suite-python-tiny-card960.json) and
[`suite-python-tiny-card640.json`](suite-python-tiny-card640.json).
The model, detector, crop preparation, batching and decoder all differ from the
production pipeline. This is evidence for a future DB adapter, not a shipped
Rust/Android speedup or an isolated runtime comparison. Title OCR remains the
current app target.

Published [PaddleOCR recognition](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)
and [detection](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html)
timings use server CPUs/GPUs and particular runtimes. They motivate experiments
but do not establish this project's Android title-update rate.
