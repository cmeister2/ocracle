# Experimental mobile-model probe

`mobile_ocr_probe.py` is a **Python experiment**, not an Android production
implementation. The immediate target is reading a supplied title crop at about
15 FPS. `line` mode runs recognition only. The supplementary `full` mode runs
fresh DB detection, sorted quadrilateral crops, and CTC recognition.

The probe does not select crops using ground truth, cache model predictions,
download weights, or modify model graphs. Every measured call includes pixel
preparation, inference, decoding, and result geometry. Loading sessions and
decoding the image file are excluded, matching the existing latency harness.
Python CPU timings do not establish Android frame rate, camera-title localization,
thermal behavior, or mobile battery use.

## Reproduce

Use the hash-pinned downloader described in
[mobile-models.md](../docs/performance/mobile-models.md). It preserves official
`inference.yml` and model bytes. Create a separate environment:

```sh
python3 -m venv /tmp/ocracle-mobile-probe-venv
/tmp/ocracle-mobile-probe-venv/bin/pip install -r scripts/mobile_ocr_probe_requirements.txt
```

Define the model/configuration paths for the candidate under examination; these
are examples, not script defaults:

```sh
PROBE_PYTHON=/tmp/ocracle-mobile-probe-venv/bin/python
PROBE_REC=.cache/performance/mobile-models/en_PP-OCRv5_mobile_rec_onnx
PROBE_DET=.cache/performance/mobile-models/PP-OCRv5_mobile_det_onnx

"$PROBE_PYTHON" scripts/mobile_ocr_probe.py \
  --make-config "$PROBE_REC/inference.yml" \
  --config-output "$PROBE_REC/probe-config.json" --language en

# Run only when the shared benchmark machine is otherwise idle.
"$PROBE_PYTHON" scripts/mobile_ocr_probe.py \
  "$PROBE_REC/inference.onnx" "$PROBE_REC/probe-config.json" \
  docs/gallery/assets/matrix-title-blur.png 20 line

"$PROBE_PYTHON" scripts/mobile_ocr_probe.py \
  "$PROBE_REC/inference.onnx" "$PROBE_REC/probe-config.json" \
  docs/performance/fixtures/divination-body.png 20 full \
  --detector "$PROBE_DET/inference.onnx" --max-side 960
```

`--max-side 512` and `--max-side 640` are explicit detector experiments. The
official YAML controls DB thresholds, unclipping, normalization, recognition
shape, and alphabet. The optional JSON sidecar records the exact YAML digest.
When downloaded provenance is present, model/YAML hashes are checked against it.
The JSON report records both model identities, configuration identities, native
runtime identity, dependency versions, batch shapes, original-coordinate
quadrilaterals/bounds, and individual timing samples.

The positional arguments and JSON fields match `performance_suite.py`. Activate
the dedicated environment so the executable's `python3` shebang selects it.
The suite's `--backend onnxruntime` setting and `ORT_DYLIB_PATH` must identify the
Python wheel's actual native library:

```sh
. /tmp/ocracle-mobile-probe-venv/bin/activate
export ORT_DYLIB_PATH="$(
  python -c 'import pathlib,onnxruntime; print(next((pathlib.Path(onnxruntime.__file__).parent/"capi").glob("libonnxruntime.so.*")))'
)"
python scripts/performance_suite.py \
  --executable scripts/mobile_ocr_probe.py \
  --backend onnxruntime --model "$PROBE_REC/inference.onnx" \
  --config "$PROBE_REC/probe-config.json" --language en \
  --workload line --iterations 20 --threads 4 \
  --label python-mobile-title-screen --output /tmp/mobile-title-screen.json
```

The suite chooses fixture workloads; use its `--sample` filters for a title-only
subset. `OCRACLE_DB_MAX_SIDE` overrides detector size and
`OCRACLE_REC_BATCH_SIZE` overrides recognition batch size (default six, with one
item for a title crop). `OCRACLE_DETECTOR_MODEL`/`OCRACLE_DETECTOR_CONFIG` choose the
detector. `OCRACLE_LAYOUT=standard_card` applies the same geometric artwork mask
as Rust, inside every timed full-image call. It is ignored for single-line crops.
There is no text-line orientation classifier or document unwarping stage.

## Source contract and differences from the Rust pipeline

Processing is adapted from PaddleX commit
[`c852371c701d83cd21955b2963834825920ffebb`](https://github.com/PaddlePaddle/PaddleX/tree/c852371c701d83cd21955b2963834825920ffebb).
That revision includes PP-OCRv6; older v3.3.0 predates its model-name defaults.

* [Detection predictor](https://github.com/PaddlePaddle/PaddleX/blob/c852371c701d83cd21955b2963834825920ffebb/paddlex/inference/models/text_detection/predictor.py)
  selects standalone maximum-side 960 for the mobile v5/v6 candidates. This
  differs from the full OCR pipeline's minimum-side 64 override. The probe
  explicitly uses maximum-side resizing and rounds each dimension to a multiple
  of 32, with OpenCV byte bilinear interpolation.
* [Detection processing](https://github.com/PaddlePaddle/PaddleX/blob/c852371c701d83cd21955b2963834825920ffebb/paddlex/inference/models/text_detection/processors.py)
  supplies BGR affine normalization and DB contour/box scoring, PyClipper
  unclipping, minimum-side filtering, and rescaling to original coordinates.
  It reads v5/v6 threshold differences from each model's official YAML.
* [Crop processing](https://github.com/PaddlePaddle/PaddleX/blob/c852371c701d83cd21955b2963834825920ffebb/paddlex/inference/pipelines/components/common/crop_image_regions.py)
  supplies min-area refitting, perspective cropping with cubic interpolation and
  replicated borders, and rotation of tall crops. DB regions go directly to
  recognition; Ocrs word grouping and its fixed box expansion do not apply.
* [Recognition processing](https://github.com/PaddlePaddle/PaddleX/blob/c852371c701d83cd21955b2963834825920ffebb/paddlex/inference/models/text_recognition/processors.py)
  supplies BGR resizing, normalized zero padding, 3200-pixel width cap, batch
  padding, and CTC decoding. Wide tensor widths are truncated after multiplying
  height by the maximum aspect ratio; this can differ by one pixel from Rust's
  configured ceiling rule. CTC ties select the first class. Confidence is the
  arithmetic mean of first-emission posteriors, unlike Rust's repeated-run peak
  geometric mean. Neither score is calibrated.

The adapted code retains the PaddlePaddle copyright notice and is distributed
under the included [Apache-2.0 license](licenses/PaddleX-APACHE-2.0.txt).
The remainder of the repository keeps its existing license.

## Tests without inference

```sh
/tmp/ocracle-mobile-probe-venv/bin/python -m unittest discover \
  -s scripts -p test_mobile_ocr_probe.py
```

For independent upstream parity checks, fetch these exact source files; tests
verify their SHA-256 before executing selected definitions. No neural models or
Paddle framework are loaded:

```sh
mkdir -p /tmp/ocracle-paddlex-reference
PROBE_SOURCE=https://raw.githubusercontent.com/PaddlePaddle/PaddleX/c852371c701d83cd21955b2963834825920ffebb
curl --fail --location "$PROBE_SOURCE/paddlex/inference/models/text_detection/processors.py" \
  -o /tmp/ocracle-paddlex-reference/detection.py
curl --fail --location "$PROBE_SOURCE/paddlex/inference/models/text_recognition/processors.py" \
  -o /tmp/ocracle-paddlex-reference/recognition.py
curl --fail --location "$PROBE_SOURCE/paddlex/inference/pipelines/components/common/crop_image_regions.py" \
  -o /tmp/ocracle-paddlex-reference/crop.py
PADDLEX_REFERENCE_DIR=/tmp/ocracle-paddlex-reference \
  /tmp/ocracle-mobile-probe-venv/bin/python -m unittest discover \
  -s scripts -p test_mobile_ocr_probe.py
```

The checks cover BGR values, resize/padding boundaries, CTC Unicode/ties/repeats,
DB scores and original-coordinate boxes, perspective crop shape/rotation,
changed configuration rejection, and fresh recognition calls. Upstream parity
compares exact detector/recognition tensors, perspective-crop pixels, DB
boxes/scores, and decoded results.
