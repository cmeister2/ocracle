# Native Paddle training feasibility

Reproducible CPU proof for the exact official PP-OCRv6 small CTC recognizer
currently used by OCRacle. It loads native pretrained weights, checks forward
parity against the deployed ONNX, appends three missing characters and performs
three gradient updates with a clean teacher anchor and frozen batch statistics.
The adapted weights export through official Paddle2ONNX and pass ONNX Runtime,
RTen and actual Rust CLI checks with all appended classes retained.

This establishes a trainable path. It does not establish improved card OCR.
Production models, language profiles and held-out tests remain unchanged.

See [source pins, commands, results and limitations](../../docs/experiments/paddle-training.md).

- [Source files](source-files.json), [checkpoint](source-checkpoint.json),
  [CPU environment](environment.json) and [wheel lock](requirements.lock).
- [Native/ONNX parity](parity.json).
- [Autograd and vocabulary smoke](smoke.json).
- [Adapted export and Rust parity](export.json).

`vendor/`, `cache/` and `runs/` are local, ignored outputs. The trained smoke
checkpoint is deliberately not distributed or registered.
