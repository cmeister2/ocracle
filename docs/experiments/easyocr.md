# EasyOCR Japanese compatibility experiment

This preserves a reproducible **best-effort experiment**, including failures.
The current practical multilingual baseline uses the separately evaluated Paddle
model. These EasyOCR exports are not registered production profiles and have not
been fine-tuned on MTG.

The exporter loads complete official pretrained CRNNs: a VGG backbone and two
BiLSTMs for generation 2, or a ResNet backbone and two BiLSTMs for generation 1.
It does not initialize an entire Japanese classifier from scratch. Source and
checkpoint metadata are pinned in `model_sources.json`; downloaded source,
weights, exported models and logs are ignored by this directory's `.gitignore`.
The upstream source uses the included [Apache 2.0 license](../../training/experimental_easyocr/LICENSE-EasyOCR).

The ONNX wrapper preserves every learned layer, replaces an equivalent adaptive
height pool with a mean, scales ocracle's `pixel/255-.5` input by two to match
EasyOCR's `[-1,1]`, and returns `[time,1,class]` log probabilities with blank class
zero. Removing unused torchvision detector imports lets recognition export use
the existing training environment. Batch size is fixed at one; width is dynamic.
Each export checks the wrapper against upstream PyTorch and compares ONNX output
at widths 128, 160 and 287. The actual Rust CLI loaded and ran both architectures.

## Evidence and limits

`corpus/cards.json` contains 90 actual Japanese print metadata records across
11 historic set cohorts plus fixed image evaluation prints. This is an alphabet
audit, not a manually transcribed recognition benchmark. It preserves Japanese
printing fields and does not substitute English Oracle text. The source bulk
snapshot and deterministic selection are recorded in the file.

`corpus/alphabet-audit-ja.json` finds 673 distinct non-whitespace reference
characters. The generation 1 alphabet misses 43 of them, with 793 occurrences;
generation 2 misses 71, with 421 occurrences. Generation 2 is selected by the
explicit frequency-based policy, but neither alphabet is complete. Notably,
generation 2 cannot emit `択` in `選択`, and neither can emit fullwidth digits.
Generation 1 also lacks common Japanese punctuation. Character presence alone
does not prove that the network recognizes its card glyph.

`fixtures/fixtures.json` describes 7 manually checked Japanese line crops and
paired severe blur, from two actual print images. It includes title, isolated
furigana, rules, reminder and flavor typography, plus an intrinsically blurry
source print. Source image hashes, crop coordinates and print URLs are recorded.
The flavor/reminder characters are upright in these prints; they are not evidence
of Latin italic coverage. Vertical Japanese and mixed furigana segmentation are
unverified.

`japanese-report.json` records all **28 actual Rust/RTen observations** from the
two models, including complete preprocessing candidates, methods, uncalibrated
confidence, image/model/config hashes, alphabet gaps and measured errors. Neither
model gets any of the severe-blur crops exactly right. This is a small stress
sample, not an estimate of language-wide accuracy. Opt appeared in the frozen
Latin V1 training corpus, so these are pretrained-model test fixtures rather than
globally unseen canonical card identities. Upstream pretraining membership is
unknown. No lookup or Oracle text replacement corrects the predictions.

## Reproduce

Use the existing training environment and its pinned requirements. From the repo
root, materialize the shared digest-verified Scryfall bulk snapshot first, then:

```sh
python3 training/experimental_easyocr/acquire_corpus.py
python3 training/export_easyocr.py --audit-only
python3 training/experimental_easyocr/acquire_fixtures.py
training/.venv/bin/python training/export_easyocr.py \
  --checkpoint japanese_g2 --allow-alphabet-gaps
training/.venv/bin/python training/export_easyocr.py \
  --checkpoint japanese_g1 --allow-alphabet-gaps
cargo build --release --locked
python3 training/experimental_easyocr/evaluate.py
```

The explicit `--allow-alphabet-gaps` flag permits an incomplete evaluation export;
it does not declare the language supported. Without it, exporting a candidate
that cannot emit every corpus character fails. SHA-256 pins are provided for both
Japanese checkpoints already acquired; other listed pretrained profiles retain
the upstream MD5 and record SHA-256 on acquisition, and require their own localized
corpus audit and image-level evaluation before admission.

Generation 1 Japanese weights are roughly 221 MB; generation 2 is roughly 17 MB.
The exporter also has pretrained candidate metadata for Latin, Simplified and
Traditional Chinese, Korean, Cyrillic, Arabic and Devanagari/Sanskrit, but those
profiles have not been evaluated by this experiment. Greek and Hebrew are not
provided by this EasyOCR checkpoint catalog.

Primary upstream references:

- [Pinned model/alphabet configuration](https://github.com/JaidedAI/EasyOCR/blob/363afb184047ce452e436f4224f3098422df872e/easyocr/config.py)
- [Recognition preprocessing and model loader](https://github.com/JaidedAI/EasyOCR/blob/363afb184047ce452e436f4224f3098422df872e/easyocr/recognition.py)
- [Generation 2 architecture](https://github.com/JaidedAI/EasyOCR/blob/363afb184047ce452e436f4224f3098422df872e/easyocr/model/vgg_model.py)
- [RTen model format and ONNX loading](https://github.com/robertknight/rten/blob/main/docs/model-formats.md)
