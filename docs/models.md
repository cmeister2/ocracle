# Neural model artifacts

The normal API is `CardOcr::open(root)?.read(&image)`. A printed language can
be supplied through `open_language`; callers do not choose a network size or
preprocessing recipe. Six evaluated Latin-script languages use the unmodified
official **PP-OCRv6 medium** recognizer and its original-image pass. Latin `la`
and best-effort Japanese/Chinese retain small pending separate evidence.

The [medium manifest](../models/paddle-v6-medium-manifest.json) and
[archived small manifest](../models/paddle-v6-manifest.json) pin upstream
revisions, graph/configuration hashes and Apache-2.0 attribution. The downloader
fetches the required artifacts. Rust preserves BGR colour, normalizes to
`[-1,1]`, resizes to height48, pads short lines to width320 and decodes native
NTC probabilities. No Python runtime is required for recognition.

The [fresh comparison](experiments/paddle-blur.md) supports medium's lower
aggregate character error while exposing clean exact-line regressions. Missing
`¡`, `„`, `ẞ`, mana symbols and incomplete layout remain limitations. These are
pretrained networks evaluated on MTG, without MTG fine-tuning. The
[Apache license](../models/LICENSE-PADDLE-APACHE-2.0.txt) is retained.
Historical weights/configurations and result bytes are preserved independently
of active language defaults; no model-size or experimental quality aliases are
registered.

`ocracle-mtg.rten` is the distributed MTG-adapted CRNN recognizer. Its alphabet,
input height, and model name are in [recognizer-config.json](../models/recognizer-config.json).
The model takes grayscale `[N,1,64,W]` floats normalized as `pixel / 255 - 0.5`
and emits `[time,N,97]` log probabilities. Class 0 is the CTC blank; the other
classes follow the configuration alphabet. Output length is `floor(W/4)+1`.
Rust uses batch size 1 and decodes the entire output sequence.

The [manifest](../models/manifest.json) pins artifact hashes and upstream provenance.
Fetch the detector and upstream reference recognizer with:

```sh
python3 training/download_models.py
```

`--checkpoint` also downloads the PyTorch checkpoint used for adaptation.
Downloads are atomic and fail if SHA-256 does not match. The detector uses the
current official Ocrs ONNX endpoint, pinned by digest. The recognizer/checkpoint
use a pinned Hugging Face repository revision. The older Hugging Face detector
is intentionally not used: its outputs did not work with the current detection
pipeline in an actual card test.

Example:

```sh
cargo run --release -- recognize docs/gallery/assets/beleren-title.png \
  --line --recognition-model models/ocracle-mtg.rten \
  --model-config models/recognizer-config.json
```

For full cards, add `--detection-model models/text-detection.onnx`. The adapted
model reads printed pixels; it does not retrieve or substitute Oracle text.
Confidence is an uncalibrated geometric mean of emitted CTC posterior peaks,
not a probability that the whole transcription is correct.

Training data, reproducible commands, damage conditions, held-out metrics, and
coverage limitations are documented in [training](training/english-crnn.md).
Several historical and special frames and actual italics are represented;
universal font support and recognition of arbitrary unreadable blur are not
established. The model is English/Latin and does not encode Magic mana symbols.

## Model license and attribution

The MTG-adapted and original Ocrs weights are licensed **CC BY-SA 4.0**, separately from the MIT Rust
and Python source. See [the complete model license](../models/LICENSE-CC-BY-SA-4.0.txt).
The distributed recognizer is an adaptation of Robert Knight's
[Ocrs recognition checkpoint](https://huggingface.co/robertknight/ocrs),
fine-tuned for this repository on MTG text and card scan crops with regular and
italic text and optical damage augmentations. Changes comprise domain-specific
CTC fine-tuning and export to RTen; the architecture/alphabet remain compatible.

The upstream checkpoint was trained on HierText and synthetic data. HierText
is by Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa
Fujii, and Michalis Raptis. Dataset/research:
[HierText](https://github.com/google-research-datasets/hiertext),
[paper](https://arxiv.org/abs/2203.15143).

Wizards of the Coast retains rights to Magic card text and imagery. The source
card scans and font programs are not distributed with these weights. This is
unofficial fan research and has no affiliation with Wizards or Scryfall.
