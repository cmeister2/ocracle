# Neural model artifacts

Default Latin-script profiles and the best-effort Japanese/Chinese profiles use
the unmodified official **PP-OCRv6 small** ONNX recognizer, downloaded by
`training/download_models.py`. Its [manifest](../models/paddle-v6-manifest.json) pins the
upstream revision, file digest, Apache-2.0 license, matching per-language
configurations and evaluation. Rust preserves BGR colour, normalizes to `[-1,1]`,
resizes to height 48 and pads short lines to width 320. It decodes the native
NTC probability output. No ONNX wrapper or Python runtime is needed for OCR.
This is a pretrained baseline evaluated on MTG, not an MTG-fine-tuned model.
The [actual Rust comparison](../training/experimental_paddle/results-v6-direct.json)
retains every prediction and failure; complete font/language coverage is unproven.
See [the full Apache license](../models/LICENSE-PADDLE-APACHE-2.0.txt).

The optional **PP-OCRv6 medium** profiles append `-medium` to the six evaluated
language IDs: `en`, `es`, `fr`, `de`, `it` and `pt`. The
[medium manifest](../models/paddle-v6-medium-manifest.json) pins its unmodified
official ONNX graph and separate configurations; the downloader fetches both
Paddle sizes. It uses the same BGR tensor and dictionary contract as small,
including the missing `¡`, `„` and `ẞ` characters. It is upstream pretrained,
without MTG fine-tuning. The [fresh comparison](experiments/paddle-blur.md)
records lower damaged-text CER alongside clean exact-match regressions and
higher compute cost. Existing defaults and historical small-model evidence
remain preserved. `la` and best-effort Japanese/Chinese profiles retain small.

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
