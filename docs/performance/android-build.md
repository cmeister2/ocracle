# Android ARM64 build validation

The release CLI and latency executable linked successfully for
`aarch64-linux-android` in both the default and `--features onnxruntime`
configurations, using NDK r30 (`30.0.16248370`), API 24, and Rust 1.99.0.
Both builds also produced the Rust library archive. The tested source is
commit `4dcea58b1160f20ad4dc170b5dce413c899d3a79`, which includes the optional
runtime backend, shared per-engine RTen inference pools, portable detector and
native regression tests. All source,
manifest and lockfile hashes stayed unchanged throughout the checks.

The [build record](android-build.json) identifies the source files, toolchain,
commands and artifact hashes; [compiler output](android-build-output.txt)
records both successful builds and the declared MSRV check. This validates
target code generation and native linking. No handset was available, so runtime
behavior, sustained latency, power and Android FPS remain unmeasured.

| Configuration | CLI bytes | Latency executable bytes |
| --- | ---: | ---: |
| Default (RTen) | 9,933,464 | 9,315,904 |
| Optional ONNX Runtime feature | 10,242,368 | 9,625,304 |

All four executables are ARM64 PIEs using `/system/bin/linker64`, with 16 KiB
load-segment alignment. Their shared dependencies are Android's `libdl.so`,
`libm.so` and `libc.so`; no separate RTen shared library is required.
[ELF inspection](android-elf.txt) records each executable independently.

Enabling the optional feature compiles the ONNX Runtime adapter; RTen remains
the default backend. Selecting ONNX Runtime also requires an application-supplied
Android `libonnxruntime.so`, loaded dynamically at runtime. Successful linking
does not validate that library loading or inference works on a device. See the
[pinned native library record](android-runtime.json) and
[device benchmark procedure](android-benchmark.md).

The installed Rust 1.94.0 toolchain also passed
`cargo +1.94.0 check --locked --all-targets --all-features` on the Linux host.
This verifies the declared MSRV against the locked dependency graph, including
the optional backend; it is separate from Android linking with Rust 1.99.0.

## Reproduction

The official [NDK download page](https://developer.android.com/ndk/downloads)
published the Linux archive as 738,633,529 bytes with SHA-1
`5107f898313790e449e87eee2183d9a20602dee9`. Both matched the downloaded file;
the independently recorded SHA-256 is in the build record.

```sh
curl --fail --location --retry 3 \
  --output /tmp/android-ndk-r30-linux.zip \
  https://dl.google.com/android/repository/android-ndk-r30-linux.zip
python3 - <<'PY'
import hashlib
from pathlib import Path
p = Path("/tmp/android-ndk-r30-linux.zip")
if p.stat().st_size != 738633529:
    raise SystemExit("NDK archive size mismatch")
with p.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha1").hexdigest()
if actual != "5107f898313790e449e87eee2183d9a20602dee9":
    raise SystemExit("NDK archive checksum mismatch")
print("NDK archive verified")
PY
```

After verification, extract into a fresh directory, then build. Save each
variant before the next command replaces Cargo's output paths:

```sh
unzip -q /tmp/android-ndk-r30-linux.zip -d /tmp
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=/tmp/android-ndk-r30/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang
mkdir -p /tmp/ocracle-android-final/default /tmp/ocracle-android-final/onnxruntime
cargo build --release --locked --target aarch64-linux-android \
  --lib --bin ocracle --example latency
cp target/aarch64-linux-android/release/ocracle \
  target/aarch64-linux-android/release/examples/latency \
  target/aarch64-linux-android/release/libocracle.rlib \
  /tmp/ocracle-android-final/default/
cargo build --release --locked --target aarch64-linux-android \
  --lib --bin ocracle --example latency --features onnxruntime
cp target/aarch64-linux-android/release/ocracle \
  target/aarch64-linux-android/release/examples/latency \
  target/aarch64-linux-android/release/libocracle.rlib \
  /tmp/ocracle-android-final/onnxruntime/
for variant in default onnxruntime; do
  for executable in ocracle latency; do
    file "/tmp/ocracle-android-final/$variant/$executable"
    /tmp/android-ndk-r30/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf \
      --file-header --program-headers --dynamic --notes \
      "/tmp/ocracle-android-final/$variant/$executable"
  done
done
cargo +1.94.0 check --locked --all-targets --all-features
```

The Rust target must already be installed. Merely running `cargo check` does not
exercise the linker. A `cargo rustc` staticlib attempt without the NDK also fails
for this dependency graph because RTen/Ocrs produce cdylibs during compilation.
The configured NDK linker resolves that missing build step.
