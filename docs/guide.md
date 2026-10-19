# Usage and development guide

Neural OCR for **Magic: The Gathering** card images: titles, type lines, printed rules, italic reminder text and flavor text. Rust performs inference locally using CTC neural networks. The default is a colour PaddleOCR baseline evaluated on printed Magic text. A separate Python workflow fine-tunes a CRNN on Magic text with deliberately blurry training images.

This is an experimental project. The target includes **all printed languages** and historic fonts, with a single language profile per card. Latin-script languages are the priority; non-Latin alphabets are best effort. Standard historic type families include Goudy Medieval, Matrix Bold, Beleren, and Plantin regular/italic; the gallery also evaluates special treatments. Support for every font, treatment and language is not yet established. Phyrexian is a separate stretch goal. Results and gaps are recorded instead of treating a font name in a configuration file as proof of recognition.

## Read a card

Requires Rust **1.94.0** or newer (edition 2024), Python 3 for downloading models, and an internet connection for that initial download. OCR itself works offline. Model files are separately distributed and are not included in the crate package.

```sh
git clone https://github.com/cmeister2/ocracle.git
cd ocracle
python3 training/download_models.py
cargo build --release

target/release/ocracle recognize card.jpg --language en
```

For an isolated text line, skip detection:

```sh
target/release/ocracle recognize docs/gallery/assets/historic-flavor-blur.png \
  --language en --line --passes all
```

`--language` selects one profile for the entire card from `models/profiles.json` and verifies SHA-256 digests of both its weights and exact configuration file. Latin-script profiles are `en`, `es`, `fr`, `de`, `it`, `pt`, and `la`; `ja`, `zhs`, and `zht` have best-effort horizontal examples. These share the upstream PP-OCRv6 small model, without MTG fine-tuning. Coverage is limited to evaluated samples, and unknown profiles return an error. Use `--model-root /path/to/ocracle` when running outside the repository. Custom weights can be selected directly with `--recognition-model` and `--model-config`.

The experimental MTG-adapted English CRNN is checked into the repository with its matching configuration; select it with `--language en-mtg`. The default Paddle model is downloaded from a pinned official source. For comparison, the downloader also fetches the original Ocrs base recognizer:

```sh
target/release/ocracle recognize card.jpg \
  --detection-model models/text-detection.onnx \
  --recognition-model models/text-recognition-base.rten
```

The JSON output includes raw text, line coordinates in the original image, confidence, the actual preprocessing method, all competing passes, and model SHA-256 hashes. `--passes original` reads the input without enhancement; `enhanced` tries enlargement, contrast stretching and sharpening; `all` retains both and selects by character posterior score. Enhancement cannot reconstruct information erased by blur.

`--layout text` (the default) scans the whole image. For conventional upright cards, `--layout standard-card` masks the illustration interior to reduce false text detections in artwork. `--layout auto` applies that mask based on aspect ratio alone; use `text` for Sagas, Adventures, split cards, full-art cards, showcase frames or arbitrary text boxes. Single-line crops always bypass masking. This geometric heuristic is not a general card-layout classifier.

**Confidence** is the geometric mean of the peak posterior for each emitted CTC character, weighted by character count across lines. It is uncalibrated and is not the probability that a line is correct. Selection by this score can favor a short, incorrect result; the alternatives remain visible. No dictionary or Scryfall/Oracle lookup replaces the observed text.

The default alphabet has 18,709 Unicode characters, including accents and curly quotation marks. It still lacks `¡`, `„`, and complete mana/tap symbol support. The experimental `en-mtg` alphabet is much narrower: English letters, numbers, ASCII punctuation and the euro sign. Card faces should be upright and reasonably flat; an isolated line or text-box crop can help difficult layouts. Vertical text, ruby grouping and right-to-left decoding are not implemented by the registered profiles.

## Library

```rust,no_run
use ocracle::{CardOcr, ModelConfig, Options};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let ocr = CardOcr::load(
        "models/text-recognition-base.rten",
        Some(Path::new("models/text-detection.onnx")),
        ModelConfig::default(),
    )?;
    let result = ocr.recognize(&image::open("card.jpg")?, Options::default())?;
    println!("{}\nscore: {:.3}", result.text, result.confidence);
    Ok(())
}
```

Models can also be ONNX files compatible with RTen. Defaults retain the original CRNN contract: grayscale `[N,1,height,W]`, pixel values `/255 - 0.5`, and `[T,N,C]` **log probabilities**, with CTC blank at zero. Colour CTC exports can configure planar RGB/BGR input, affine normalization, right padding, width rounding, NTC output and probabilities. Rust processes one line per batch and uses triangle resizing; this can differ from an upstream OpenCV interpolation pipeline. Each configuration supplies the exact checkpoint alphabet, `input_height` and `architecture` for method attribution. Changing the alphabet without retraining changes what classes mean. RTen 0.26 matches `ocrs` 0.13's detector integration.

## Training and evaluation

See [training](training/english-crnn.md) for reproducible acquisition, card-identity splits, augmentation, training, export and measured results, and the [font coverage audit](font-coverage.md) for the broader inventory and gaps. Real scan crops expose the historical proprietary letterforms without redistributing proprietary font files. Open fonts supply additional regular and italic synthetic examples; they are explicitly identified as substitutes, not the original Magic fonts.

The [language inventory](language-coverage.md) tracks verified print languages and their transcription requirements. Language profiles are recorded separately from working models: adding a profile or initializing a Unicode classifier does not enable recognition of that language.

An [experimental shared Latin model](training/latin-v1.md) adds accented classes for French, German, Italian, Portuguese and Spanish. Its [independent printed-title evaluation](../training/latin-eval/README.md) exposed frequent accent errors, including confidently incorrect results. The frozen weights and raw failures are retained for comparison; this experiment is not a ready language profile.

Blur is part of training: strong Gaussian blur, directional motion blur, downsampling, JPEG damage and combinations. Clean and damaged variants of a card stay in the same split. The gallery's card identities are excluded from specialized training. Evaluation must report clean and damaged cases separately; a model can still fail badly on severe blur.

The [fresh blur development corpus](evaluation/blur-development.md) adds 64 independently reviewed printed lines from sixteen further card identities, including 36 italic lines, under seven fixed image conditions. It supports development comparisons and is kept out of training.

## Simple gallery

The static [live gallery](https://cmeister2.github.io/ocracle/) ([source](gallery/index.html)) pairs real printed crops with actual Rust inference, method and confidence. It includes original and severely blurred versions of historic titles, rules, flavor and reminder text, plus manual printed transcriptions and character error rates. Source card links and image provenance are retained.

```sh
python3 scripts/gallery_export.py \
  --recognition-model models/ocracle-mtg.rten \
  --model-config models/recognizer-config.json
python3 scripts/gallery_export.py --check
python3 -m http.server 8000 --directory docs/gallery
```

Open `http://localhost:8000`. The GitHub Pages workflow publishes `docs/gallery` from the default branch when Pages uses **GitHub Actions** as its source. The model selector compares specialized and base checkpoints on the same crops; predictions and scores are never hand-authored.

## Development and releases

```sh
cargo check --locked
cargo test --locked
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo +1.94.0 check --locked
cargo +1.94.0 test --locked
cargo +1.94.0 clippy --all-targets --all-features -- -D warnings
cargo msrv verify
cargo llvm-cov --all-targets --all-features
npm ci
npm test
```

Release tooling needs Node **24.15+**. CI separately checks stable, MSRV and coverage, and tests release-note rendering through semantic-release's actual dependency graph. The checked-in crate and lockfile version remains `0.0.0`; `scripts/prepare-release.mjs` injects matching release versions. Plain SemVer tags and Conventional Commits drive GitHub/crates.io releases on main/master pushes. Set the repository's `CARGO_REGISTRY_TOKEN` secret before publishing. `feat:` triggers a minor release, `fix:`/`perf:` a patch, and a `BREAKING CHANGE:` footer a major release.

Submit changes through pull requests; `main` requires the aggregate **Success** check. It passes only when every required CI job succeeds, and also gates release publication. Pull requests can be stacked against an earlier branch and retargeted after its parent merges.

Rust code is MIT licensed. Downloaded model weights, open fonts, and card imagery retain their separate licenses and attribution. Magic: The Gathering is owned by Wizards of the Coast. This project is independent; sample card text/images are not covered by this project's code license. See [model provenance](models.md) and each gallery source link.
