# Measure title or whole-card OCR on an Android device

These commands prepare a native command-line benchmark for a physical ARM64
Android device, API 24 or newer. They do not create an Android app. No device was
available during this investigation: [NDK linking](android-build.md) and
[native-library inspection](android-runtime.json) are build evidence only.
The approximately 15 FPS target applies to title OCR. The default commands below
use an upright title crop and bypass detection. Full-card mode is a separate,
more expensive workload.

Run from the repository root in Bash on Linux, with Rust's
`aarch64-linux-android` target, the verified NDK from `android-build.md`, Python 3,
curl, and [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools).
The following commands use `adb -d` to select one physical USB device; use
`adb -s SERIAL` throughout if more than one is attached. Enable USB debugging
and authorize the development computer as described in the
[ADB documentation](https://developer.android.com/tools/adb).

## Build both release variants

The default build uses RTen. The optional build adds dynamically loaded ONNX
Runtime CPU inference. Both use the same model weights, image processing and CTC
decoder. This does not enable NNAPI, XNNPACK, Vulkan or a GPU provider.

```bash
set -euo pipefail
export OCRACLE_PACKAGE=/tmp/ocracle-android-package
export OCRACLE_RESULTS=/tmp/ocracle-android-results
export OCRACLE_MODE=line
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=/tmp/android-ndk-r30/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang
mkdir -p "$OCRACLE_PACKAGE"
mkdir "$OCRACLE_RESULTS"
git rev-parse HEAD > "$OCRACLE_RESULTS/source-commit.txt"
git diff --binary HEAD > "$OCRACLE_RESULTS/source.patch"
git ls-files -z src Cargo.toml Cargo.lock examples/latency.rs |
  xargs -0 sha256sum > "$OCRACLE_RESULTS/source.sha256"
rustc -vV > "$OCRACLE_RESULTS/rustc.txt"

CARGO_TARGET_DIR=/tmp/ocracle-android-rten \
  cargo build --release --locked --no-default-features \
  --target aarch64-linux-android --lib --example latency
cp /tmp/ocracle-android-rten/aarch64-linux-android/release/examples/latency \
  "$OCRACLE_PACKAGE/latency-rten"

CARGO_TARGET_DIR=/tmp/ocracle-android-ort \
  cargo build --release --locked --no-default-features --features onnxruntime \
  --target aarch64-linux-android --lib --example latency
cp /tmp/ocracle-android-ort/aarch64-linux-android/release/examples/latency \
  "$OCRACLE_PACKAGE/latency-onnxruntime"
```

Commit new source files before recording the source identity; `git diff` and
`git ls-files` do not capture untracked files. Keep the compiler logs with the
results. The two target directories prevent feature-specific artifacts from
overwriting each other.

## Supply the Android native runtime and identical inputs

ONNX Runtime's official [Android installation instructions](https://onnxruntime.ai/docs/install/#install-on-android)
distribute native libraries in the `onnxruntime-android` AAR on Maven Central.
The pin below is **1.31.0**, rather than `latest.release`. Its
[published SHA-1](https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.31.0/onnxruntime-android-1.31.0.aar.sha1)
was verified and its SHA-256 independently recorded in `android-runtime.json`.
The ARM64 library has a minimum API of 24 and only Android system dependencies;
the Java JNI companion is unnecessary for this Rust C API client.

```bash
curl --fail --location --retry 3 \
  --output /tmp/onnxruntime-android-1.31.0.aar \
  https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.31.0/onnxruntime-android-1.31.0.aar
python3 - <<'PY'
import hashlib, os, zipfile
from pathlib import Path
p = Path("/tmp/onnxruntime-android-1.31.0.aar")
with p.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
if actual != "72683787a80d0608a3121b967d8e1239aef48286826661c953211626690540a7":
    raise SystemExit("Android ONNX Runtime archive checksum mismatch")
with zipfile.ZipFile(p) as archive:
    library = archive.read("jni/arm64-v8a/libonnxruntime.so")
Path(os.environ["OCRACLE_PACKAGE"], "libonnxruntime.so").write_bytes(library)
PY

cp models/ppocr-v6-medium.onnx "$OCRACLE_PACKAGE/recognizer.onnx"
cp models/paddle-v6-medium/en.json "$OCRACLE_PACKAGE/recognizer.json"
cp models/text-detection.onnx "$OCRACLE_PACKAGE/detector.onnx"
cp docs/gallery/assets/beleren-title.png "$OCRACLE_PACKAGE/input.png"
sha256sum "$OCRACLE_PACKAGE/"* > "$OCRACLE_RESULTS/artifacts.sha256"
cp docs/performance/fixtures/manifest.json "$OCRACLE_RESULTS/fixtures.json"
```

Check `input.png` against `line-beleren-title-clean` in that manifest. For a
separate whole-card run, set `OCRACLE_MODE=full` and copy
`.cache/gallery/m15-52.png` to `input.png`; use
`scripts/performance_suite.py --prepare --download-sources` if the pinned scan is
missing, and check it against `card-divination`. To evaluate the small recognizer, pair
`models/ppocr-v6-small.onnx` with `models/paddle-v6/en.json` and use a separate
result directory. Preserve the original detector when comparing runtimes;
changing its resolution is a separate accuracy experiment. Do not copy a Linux
Python wheel's native library onto Android.

## Deploy and record the device

```bash
adb devices -l
adb -d shell getprop ro.product.cpu.abilist
adb -d shell getprop ro.build.version.sdk
adb -d shell mkdir -p /data/local/tmp/ocracle-bench
adb -d push "$OCRACLE_PACKAGE/." /data/local/tmp/ocracle-bench/
adb -d shell chmod 755 /data/local/tmp/ocracle-bench/latency-rten \
  /data/local/tmp/ocracle-bench/latency-onnxruntime
adb -d shell 'cd /data/local/tmp/ocracle-bench && sha256sum *' \
  > "$OCRACLE_RESULTS/device-artifacts.sha256"
adb version > "$OCRACLE_RESULTS/adb.txt"
adb -d shell 'for p in ro.product.manufacturer ro.product.model ro.soc.model ro.hardware ro.build.fingerprint ro.build.version.sdk ro.product.cpu.abilist; do echo "$p=$(getprop "$p")"; done' \
  > "$OCRACLE_RESULTS/device.txt"
adb -d shell cat /proc/cpuinfo > "$OCRACLE_RESULTS/cpuinfo.txt"
```

Require `arm64-v8a` and API 24 or newer; compare host/device hashes by basename.
Record ambient conditions, case/cooling, screen state, battery level, charging
state and background workload alongside the results. Keep those conditions
consistent and let the device return to the same thermal state between paired
runs. Do not disable thermal management.

## Sweep one, two and four threads

The benchmark loads once, times its first OCR call, warms three more times, then
records every measured call. Warm title times include preprocessing, recognition,
CTC and result creation; full mode also includes detection and line extraction.
Model loading and image decoding are
outside warm timings. `RTEN_NUM_THREADS` controls RTen; the latency example also
passes this value explicitly to the optional ONNX Runtime session.

```bash
capture_state() {
  adb -d shell dumpsys thermalservice > "$OCRACLE_RESULTS/$1-thermal.txt" 2>&1 || true
  adb -d shell dumpsys battery > "$OCRACLE_RESULTS/$1-battery.txt" 2>&1 || true
}
run_ocr() {
  # Arguments are controlled backend names and positive integer counts.
  adb -d shell "cd /data/local/tmp/ocracle-bench &&
    RTEN_NUM_THREADS=$2 OCRACLE_BACKEND=$1 OCRACLE_LAYOUT=text \
    OCRACLE_DETECTOR_MODEL=/data/local/tmp/ocracle-bench/detector.onnx \
    ORT_DYLIB_PATH=/data/local/tmp/ocracle-bench/libonnxruntime.so \
    ./latency-$1 recognizer.onnx recognizer.json input.png $3 $OCRACLE_MODE"
}
for backend in rten onnxruntime; do
  for threads in 1 2 4; do
    label="$backend-t$threads"
    capture_state "$label-before"
    run_ocr "$backend" "$threads" 20 \
      > "$OCRACLE_RESULTS/$label.json" 2> "$OCRACLE_RESULTS/$label.stderr"
    capture_state "$label-after"
  done
done
```

Use a fresh results directory for each repetition. For a paired comparison,
repeat the sweep with reversed runtime order after cooling. Inspect raw text,
line bounds, model hashes, backend, thread count and
image dimensions and `single_line` before comparing timings. `text` matches the default public
pipeline; repeat full mode with `OCRACLE_LAYOUT=standard_card` only as a separately labeled
experiment on appropriate upright standard cards.

## Sustain inference and observe thermal behavior

Choose a backend/thread count after the sweep. Estimate enough iterations for
ten minutes using its measured median, then run them in **one process**. This
estimate sets run length; report the actual sum of measured samples afterward.

```bash
backend=onnxruntime
threads=2
iterations=$(python3 - "$OCRACLE_RESULTS/$backend-t$threads.json" <<'PY'
import json, math, sys
p = json.load(open(sys.argv[1]))
print(max(20, math.ceil(600000 / p["median_ms"])))
PY
)
capture_state sustained-before
date -u +%FT%TZ > "$OCRACLE_RESULTS/sustained-start.txt"
run_ocr "$backend" "$threads" "$iterations" \
  > "$OCRACLE_RESULTS/sustained.json" 2> "$OCRACLE_RESULTS/sustained.stderr"
date -u +%FT%TZ > "$OCRACLE_RESULTS/sustained-end.txt"
capture_state sustained-after
```

While that runs, a second terminal can log thermal state every ten seconds;
stop the monitor with Ctrl-C after the benchmark finishes:

```bash
while true; do
  date -u +%FT%TZ
  adb -d shell dumpsys thermalservice
  sleep 10
done > /tmp/ocracle-android-results/thermal-timeline.txt 2>&1
```

Keep that log and state whether polling was enabled; it adds measurement work.
`dumpsys` availability and sensor visibility depend on the device. Record an
unavailable service as unavailable. Battery temperature is not a substitute for
SoC thermal status. Android's [Thermal API documentation](https://developer.android.com/games/optimize/adpf/thermal)
explains thermal throttling and device-dependent status support.

Compare the first and last 20% of `samples_ms`, and report overall and late-run
median/p95, measured duration, text errors, and available thermal observations.
Repeat with the other runtime under matched conditions, then with the other two
whole-card fixtures. Preserve failed runs and stderr.

These are repeated still-image OCR measurements. A warm median below 66.7 ms on
one crop does not establish 15 fresh whole-card frames/second. Even sustained
whole-card results exclude camera delivery, YUV conversion, rectification,
tracking, scheduling, rendering and power measurement. Those require subsequent
application/device work; a responsive preview or reused OCR result is not a new
OCR frame.
