# Native PP-OCRv6 small training path

Native CPU adaptation is feasible for the exact PP-OCRv6 small recognizer used
by OCRacle. The official training checkpoint reproduces our deployed ONNX at
the same input tensors, and the CTC branch supports gradients and append-only
vocabulary expansion. The adapted weights also export through official
Paddle2ONNX and run in the actual Rust CLI. This experiment proves the machinery;
it does not train or select an improved MTG recognizer.

The implementation and frozen evidence are in
[training/experimental_paddle_training](../../training/experimental_paddle_training/README.md).
Production models, language profiles and all held-out results are unchanged.

## Exact source and environment

The upstream source is PaddleOCR commit
[`dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`](https://github.com/PaddlePaddle/PaddleOCR/tree/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf).
Only nine required/reference source files are downloaded, including its Apache
2.0 license, original configuration and dictionary. Every file is pinned by
byte count and SHA-256 in
[source-files.json](../../training/experimental_paddle_training/source-files.json).
The native LCNetV4 backbone, LightSVTR sequence encoder, CTC classifier and CTC
loss use the unmodified official modules. No PyTorch translation is involved.
The [official configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml)
specifies the small backbone and a 120-dimensional, two-block LightSVTR neck.

The [official model table](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/module_usage/text_recognition.en.md)
links the native training checkpoint. Its mutable provider URL is guarded by
an exact size and digest; an upstream replacement fails verification:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| Official native `.pdparams` | 124,912,348 | `25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825` |
| Deployed official ONNX | 21,159,378 | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |

The native checkpoint has 422 state entries. Strict mapping loads all 338 CTC
branch entries, including batch-normalization statistics. The remaining 84
entries belong only to the auxiliary NRTR branch and its projection. Any
other unused entry, missing parameter or mismatched shape fails loading.
The upstream [PP-OCRv6 description](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
describes CTC inference with auxiliary NRTR training. The checkpoint/source are
used under Apache 2.0; fetched source retains its license.

The observed environment is CPython 3.12.3 on Linux x86_64, PaddlePaddle 3.3.1
CPU and ONNX Runtime 1.31.0. The CPU wheel itself is 194,839,332 bytes.
[requirements.lock](../../training/experimental_paddle_training/requirements.lock)
pins every installed dependency and wheel hash; it targets this platform and
Python ABI. [environment.json](../../training/experimental_paddle_training/environment.json)
records the exact wheels, URLs, hashes and sizes. Combined new environment, exporter,
native checkpoint and source downloads are below 400 MB, excluding the existing
ONNX. GPU setup has not been tested.

## Reproduce

Run from the repository root. Use a separate environment from the Torch CRNN
experiments. The existing official ONNX must be present at
`models/ppocr-v6-small.onnx`; the repository's normal model bootstrap supplies it.

```sh
python3.12 -m venv /tmp/ocracle-paddle-training-venv
/tmp/ocracle-paddle-training-venv/bin/python -m pip install --require-hashes -r training/experimental_paddle_training/requirements.lock
python3 training/download_models.py
/tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/bootstrap.py
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/parity.py
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python -m unittest discover -s training/experimental_paddle_training -p 'test_*.py'
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/smoke.py
```

To reproduce the additional adapted-model export proof, install the separately
pinned exporter wheels, build the existing isolated Rust raw-tensor probe and
the main CLI, then export. Build commands use the repository's existing locked
Rust dependencies; they do not change language profiles or model defaults.

```sh
/tmp/ocracle-paddle-training-venv/bin/python -m pip install --require-hashes -r training/experimental_paddle_training/requirements-export.lock
cargo build --release --locked --manifest-path training/experimental_paddle/runner/Cargo.toml --target-dir /tmp/ocracle-paddle-target
cargo build --release --locked
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/export.py --runner /tmp/ocracle-paddle-target/release/paddle-rten-probe --cli target/release/ocracle
```

The smoke test additionally needs Debian/Ubuntu's `fonts-dejavu-extra` package
(observed version `2.37-8`).
It checks the exact SHA-256 of `DejaVuSerif-Italic.ttf` before rendering. The
DejaVu/Bitstream Vera font license and path are recorded in the report; no font
binary is redistributed. Install the package with your usual package manager
if it is absent. Different font bytes deliberately stop the smoke instead of
silently changing its input. The glyphs used by this smoke are present in the
pinned font.

The scripts overwrite their own `parity.json`, `smoke.json` and `export.json` when rerun.
Native source/checkpoint and smoke weights stay in ignored `vendor/`, `cache/`
and `runs/` directories. The three-step smoke is capped at three updates. It
saves an unregistered native `.pdparams` plus `alphabet.txt`, then strictly
reloads those weights and requires identical native inference. Numerical
results can differ between CPU kernels; the declared tolerances are the
reproduction contract, not a promise of identical serialization across systems.

## Forward parity before any gradient update

The official dictionary contains 18,708 characters. Appending the configured
space produces 18,709 characters; CTC blank at index zero makes 18,710 output
classes. The native dictionary must match the deployed model configuration
character for character before loading succeeds.

[parity.json](../../training/experimental_paddle_training/parity.json) records
four identical-tensor comparisons with the deployed ONNX. They cover batches
of one/two and widths 320/513 at height 48. All timestep argmax indices match:

| Input | Shape out | Maximum absolute probability error |
| --- | --- | ---: |
| Zero, batch 1, width 320 | `1×40×18710` | `1.19e-7` |
| Random, batch 1, width 320 | `1×40×18710` | `2.06e-5` |
| Random, batch 1, width 513 | `1×64×18710` | `3.72e-5` |
| Random, batch 2, width 320 | `2×40×18710` | `1.93e-5` |

The pass threshold is `5e-5` plus identical argmax at every timestep. This is
same-weight numerical agreement, not bit-for-bit equivalence or recognition
accuracy evidence. Random probes exercise all three colour channels. Both
engines consume exactly the same float32 BGR tensors, so the comparison does
not depend on independently implemented image resizing.

## The training-mode distinction

Upstream [LCNetV4.forward](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/ppocr/modeling/backbones/rec_lcnetv4.py#L622)
uses adaptive pooling to a fixed width of 40 in training mode, and ordinary
pooling with dynamic width in evaluation mode. Switching the whole model to
training mode would change the graph whose output just passed parity.

This proof keeps every module in evaluation mode, including inference pooling,
dropout and batch normalization, while ordinary autograd remains enabled.
The final classifier's linear projection is called explicitly to obtain raw
logits. The unmodified official CTC loss accepts those logits, uses the complete
output sequence and applies its own CTC computation. It does not receive
already-softmaxed probabilities. This path adapts the deployed CTC branch and
does not reproduce the original MultiHead/NRTR training recipe.

## Vocabulary and gradient proof

The official alphabet cannot emit `¡`, `„` or `ẞ`. The experiment appends them
at class indices 18,710, 18,711 and 18,712, respectively. Blank and every old
character retain their original indices. All old classifier columns/biases
are copied exactly; new columns start from `!`, `“` and `ß` prototypes with a
lower bias. This initialization alone is not recognition of the new glyphs.
Changing the matrix width introduces only `3.82e-6` maximum old-logit rounding
error before any update. Softmax probabilities can change because new classes
enter the denominator.

The only training text is freshly rendered procedural text, not card labels:
an italic `¡Ähre! „Été“ ẞ` with Gaussian blur and downsampling, plus a separate
clean English sentence for the frozen teacher. Colour-preserving inputs use
BGR, height 48, ceiling width, normalization `2/255 × pixel − 1`, and normalized
zero padding to at least width 320. PIL bilinear resizing is used for this
small proof; deployment uses image-crate triangle resizing.

Three Adam updates train the classifier at `1e-4` and LightSVTR/upper LCNet block
at `1e-5`, with global gradient clipping at four. The objective is CTC plus
`0.5 × KL(teacher || student)` on the identical clean anchor. Student probabilities
over the old rows remain part of the full expanded softmax; renormalizing only
old rows would hide new-class mass. This is a preservation mechanism, not a
demonstration of preserved clean accuracy over a real corpus.

[smoke.json](../../training/experimental_paddle_training/smoke.json) records:

- CTC loss `43.83 → 39.47 → 35.56` in three updates.
- Finite, nonzero gradients and parameter changes in all three intended groups.
- All 116 batch-normalization mean/variance tensors unchanged.
- Exact inference agreement after strict save/reload of the expanded model.
- Seeds, input tensor hashes, font provenance, implementation hashes and source
  checkpoint/parity bindings.

The three small tests check append-only blank/index preservation, invalid
expansion rejection without mutation, and BGR/normalization/padding contracts.

## Adapted ONNX, RTen and CLI proof

The export uses official
[Paddle2ONNX v2.1.0](https://github.com/PaddlePaddle/Paddle2ONNX/blob/c8b5048c3a0903986bd3ec1cce2af9915b391c49/README_en.md),
release commit `c8b5048c3a0903986bd3ec1cce2af9915b391c49`, with ONNX 1.17.0.
[environment-export.json](../../training/experimental_paddle_training/environment-export.json)
and [requirements-export.lock](../../training/experimental_paddle_training/requirements-export.lock)
pin the additional three wheels (20,209,169 bytes). Paddle's JIT saves a PIR
graph and parameters using symbolic batch/width inputs; official Paddle2ONNX
converts them to opset 17. Optional optimizer paths are disabled to avoid
implicit package downloads. No custom ONNX graph surgery or weight remapping
is needed.

[export.json](../../training/experimental_paddle_training/export.json) binds
the exact three-step smoke checkpoint to the exported 21,331,267-byte ONNX
(`2a8461cf8941eca556a2069e13030fc20c1c66b9287d1875d217d6fa6229b9fe`).
All 18,713 classes remain, including blank and the three appended classes with
their original recorded indices. Native/ONNX comparisons check every output
class, with additional last-three-column error reporting. Five probes cover
widths 320/513, a batch of two, a clean procedural line and a blurred italic
line. Four batch-one probes also run through RTen 0.26.0 directly, without an
RTen conversion or model wrapper:

| Comparison | Probes | Maximum absolute probability error | Timestep argmax |
| --- | ---: | ---: | --- |
| Adapted native → exported ONNX Runtime | 5 | `3.10e-5` | Identical everywhere |
| Exported ONNX Runtime → RTen | 4 | `5.78e-6` | Identical everywhere |

Two procedural PNGs additionally run through a snapshotted actual Rust CLI
using an experimental configuration with the expanded dictionary. Both text
results match native/ONNX/raw RTen exactly; their model/configuration and
executable hashes are retained. The PNGs have height 48 before inference, so
this end-to-end graph check avoids differences between independent resize
implementations. Dynamic width produces 40/64 timesteps at widths 320/513.
The isolated Rust probe is batch one; batch-two agreement is checked only in
native Paddle and ONNX Runtime.

This establishes deployment compatibility of these smoke weights, not good
recognition of the new characters. In all four implementations the expanded
blurred line still reads `Ähre! Été“ ß`, dropping/substituting the newly added
glyphs after only three updates. No MTG accuracy evaluation or model promotion
is inferred from numerical parity.

## Next experiment boundary

An actual MTG adaptation run still needs a frozen development corpus with
visually reviewed physical body/italic/accent crops, canonical card identity
splits, blur variants and clean anchors. Validation should select a checkpoint
before any existing held-out evaluation is inspected. Vocabulary expansion
also needs genuine examples of each new glyph; synthetic exposure alone cannot
establish physical print coverage.

The adapted native model's ONNX export and RTen compatibility are demonstrated
for this pinned architecture, exporter and checkpoint. Different architectures
or exporter versions require their own compatibility checks. No adapted model
is distributed or registered, and this smoke supplies no claim
about improved historic fonts, italics, blur robustness or language coverage.
The broader [candidate comparison](paddle-candidates.md) is a separate question
from this exact-small-model trainability proof.
