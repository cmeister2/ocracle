# ocracle

Neural OCR for **Magic: The Gathering** card images: titles, type lines, printed rules, italic reminder text and flavor text. Rust performs inference locally using a convolutional recurrent neural network (CRNN) and CTC decoding. A separate Python workflow fine-tunes that network on Magic text with deliberately blurry training images.

This is a new experimental project. The target includes **all printed languages** and historic fonts, with a single language profile per card. The current distributed checkpoint is an **English baseline**. Standard historic type families include Goudy Medieval, Matrix Bold, Beleren, and Plantin regular/italic; the gallery also evaluates special treatments. Support for every font, treatment and language is not yet established. Phyrexian is a separate stretch goal. Results and gaps are recorded instead of treating a font name in a configuration file as proof of recognition.

## Read a card

Requires Rust **1.94.0** or newer (edition 2024), Python 3 for downloading models, and an internet connection for that initial download. OCR itself works offline. Model files are separately distributed and are not included in the crate package.

```sh
git clone https://github.com/cmeister2/ocracle.git
cd ocracle
python3 training/download_models.py
cargo build --release

target/release/ocracle recognize card.jpg \
  --detection-model models/text-detection.onnx \
  --recognition-model models/ocracle-mtg.rten \
  --model-config models/recognizer-config.json
```

For an isolated text line, skip detection:

```sh
target/release/ocracle recognize docs/gallery/assets/historic-flavor-blur.png \
  --recognition-model models/ocracle-mtg.rten \
  --model-config models/recognizer-config.json \
  --line --passes all
```

The specialized checkpoint is checked into the repository with its matching configuration. For comparison, the downloader also fetches the original base recognizer:

```sh
target/release/ocracle recognize card.jpg \
  --detection-model models/text-detection.onnx \
  --recognition-model models/text-recognition-base.rten
```

The JSON output includes raw text, line coordinates in the original image, confidence, the actual preprocessing method, all competing passes, and model SHA-256 hashes. `--passes original` reads the input without enhancement; `enhanced` tries enlargement, contrast stretching and sharpening; `all` retains both and selects by character posterior score. Enhancement cannot reconstruct information erased by blur.

`--layout text` (the default) scans the whole image. For conventional upright cards, `--layout standard-card` masks the illustration interior to reduce false text detections in artwork. `--layout auto` applies that mask based on aspect ratio alone; use `text` for Sagas, Adventures, split cards, full-art cards, showcase frames or arbitrary text boxes. Single-line crops always bypass masking. This geometric heuristic is not a general card-layout classifier.

**Confidence** is the geometric mean of the peak posterior for each emitted CTC character, weighted by character count across lines. It is uncalibrated and is not the probability that a line is correct. Selection by this score can favor a short, incorrect result; the alternatives remain visible. No dictionary or Scryfall/Oracle lookup replaces the observed text.

The initial alphabet supports English letters, numbers, ASCII punctuation and the euro sign. Curly quotation marks, accented letters, other scripts and pictographic mana/tap symbols need expanded training/alphabet support; the recognizer does not yet transcribe these reliably. Card faces should be upright and reasonably flat; an isolated line or text-box crop can help difficult layouts.

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

Models can also be ONNX files compatible with RTen. The recognition contract is grayscale `[N,1,64,W]`, pixel values `/255 - 0.5`, and `[T,N,C]` **log probabilities**, with CTC blank at zero. The configuration supplies the exact checkpoint alphabet; changing it without retraining changes what classes mean. RTen 0.26 is used to match `ocrs` 0.13's detector integration.

## Training and evaluation

See [training](training/README.md) for reproducible acquisition, card-identity splits, augmentation, training, export and measured results, and the [font coverage audit](docs/font-coverage.md) for the broader inventory and gaps. Real scan crops expose the historical proprietary letterforms without redistributing proprietary font files. Open fonts supply additional regular and italic synthetic examples; they are explicitly identified as substitutes, not the original Magic fonts.

The [language inventory](docs/language-coverage.md) tracks verified print languages and their transcription requirements. Language profiles are recorded separately from working models: adding a profile or initializing a Unicode classifier does not enable recognition of that language.

Blur is part of training: strong Gaussian blur, directional motion blur, downsampling, JPEG damage and combinations. Clean and damaged variants of a card stay in the same split. The gallery's card identities are excluded from specialized training. Evaluation must report clean and damaged cases separately; a model can still fail badly on severe blur.

## Simple gallery

The static [gallery](docs/gallery/index.html) pairs real printed crops with actual Rust inference, method and confidence. It includes original and severely blurred versions of historic titles, rules, flavor and reminder text, plus manual printed transcriptions and character error rates. Source card links and image provenance are retained.

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

Rust code is MIT licensed. Downloaded model weights, open fonts, and card imagery retain their separate licenses and attribution. Magic: The Gathering is owned by Wizards of the Coast. This project is independent; sample card text/images are not covered by this project's code license. See [model provenance](models/README.md) and each gallery source link.
