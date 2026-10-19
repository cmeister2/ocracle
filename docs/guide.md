# Usage and development guide

Neural OCR for **Magic: The Gathering** card images: titles, type lines, printed rules, italic reminder text and flavor text. Rust performs inference locally using CTC neural networks. The default is a colour PaddleOCR baseline evaluated on printed Magic text. A separate Python workflow fine-tunes a CRNN on Magic text with deliberately blurry training images.

This is an experimental project. The target includes **all printed languages** and historic fonts, with a single language profile per card. Latin-script languages are the priority; non-Latin alphabets are best effort. Standard historic type families include Goudy Medieval, Matrix Bold, Beleren, and Plantin regular/italic; the gallery also evaluates special treatments. Support for every font, treatment and language is not yet established. Phyrexian is a separate stretch goal. Results and gaps are recorded instead of treating a font name in a configuration file as proof of recognition.

## Read a card

Requires Rust **1.94.0** or newer (edition 2024), Python 3 for the initial model download, and an internet connection for that download. Recognition then runs offline.

```sh
python3 training/download_models.py
cargo run --release -- recognize card.jpg
```

English is the default. For a French printing, add `--language fr`. Supported language codes are `en`, `es`, `fr`, `de`, `it`, `pt`, `la`, `ja`, `zhs` and `zht`; non-Latin coverage is best effort. One language applies to the whole image. Unsupported languages return an error rather than silently falling back to English.

The library chooses its recognizer and preprocessing. There are no model-size or quality profiles to select. Use `--model-root /path/to/ocracle` when the downloaded models are outside the current directory. `--line` identifies an already cropped text line and skips detection.

Output is JSON containing text, original-image line coordinates, the method, an uncalibrated confidence score, retained candidates and actual model identities. Scores are not probabilities of an exact transcription. Predictions read printed pixels; no card-name or Oracle lookup replaces them.

## Library

```rust,no_run
use ocracle::CardOcr;

fn main() -> anyhow::Result<()> {
    let ocr = CardOcr::open(".")?;
    let result = ocr.read(&image::open("card.jpg")?)?;
    println!("{}", result.text);
    Ok(())
}
```

Load once and reuse the recognizer for multiple images. `CardOcr::open_language("fr", ".")` sets the printed language without choosing a model or image-processing recipe.

The default uses PP-OCRv6 medium for the six evaluated Latin-script languages and reads the original pixels in one pass. On the [fresh development comparison](experiments/paddle-blur.md), this gives 314/448 exact lines and 2.78% character error, compared with 308/448 and 2.79% when confidence selects between original and enhanced medium passes. Small remains the internal recognizer for Latin `la` and best-effort Japanese/Chinese, which lack evidence for changing models.

This is a choice by aggregate accuracy, not a guarantee for every crop. Medium's clean exactness is lower than small's in that comparison, and it can drop a French diacritic even at high confidence. These failures remain in the gallery and raw reports. Strong blur can erase information, and curved lettering, symbols, vertical text and whole-card layout remain incomplete. The [coverage inventory](font-coverage.md) records the gaps.

Advanced model-loading and preprocessing APIs remain available for reproducing research, rather than as required application configuration. Their reports retain original weights, configuration hashes and results independently of the current default.

## Training and evaluation

See [training](training/english-crnn.md) for reproducible acquisition, card-identity splits, augmentation, training, export and measured results, and the [font coverage audit](font-coverage.md) for the broader inventory and gaps. Real scan crops expose the historical proprietary letterforms without redistributing proprietary font files. Open fonts supply additional regular and italic synthetic examples; they are explicitly identified as substitutes, not the original Magic fonts.

The [language inventory](language-coverage.md) tracks verified print languages and their transcription requirements. Language profiles are recorded separately from working models: adding a profile or initializing a Unicode classifier does not enable recognition of that language.

An [experimental shared Latin model](training/latin-v1.md) adds accented classes for French, German, Italian, Portuguese and Spanish. Its [independent printed-title evaluation](../training/latin-eval/README.md) exposed frequent accent errors, including confidently incorrect results. The frozen weights and raw failures are retained for comparison; this experiment is not a ready language profile.

Blur is part of training: strong Gaussian blur, directional motion blur, downsampling, JPEG damage and combinations. Clean and damaged variants of a card stay in the same split. The gallery's card identities are excluded from specialized training. Evaluation must report clean and damaged cases separately; a model can still fail badly on severe blur.

The [fresh blur development corpus](evaluation/blur-development.md) adds 64 independently reviewed printed lines from sixteen further card identities, including 36 italic lines, under seven fixed image conditions. It supports development comparisons and is kept out of training.

## Simple gallery

The static [live gallery](https://cmeister2.github.io/ocracle/) ([source](gallery/index.html)) opens with eight fixed examples and requires no selections. Flavor and reminder italics come first, followed by historic titles and an accented French title. Every example shows both its original and severely blurred crop, actual Rust OCR, method and confidence. The selected default recognizer supplies all displayed predictions; source links, literal references and errors remain inspectable.

```sh
python3 scripts/gallery_export.py --check
python3 -m http.server 8000 --directory docs/gallery
```

Open `http://localhost:8000`. The GitHub Pages workflow publishes `docs/gallery` from the default branch when Pages uses **GitHub Actions** as its source. Complete fixture and result JSON files retain the broader historical model comparisons; displayed predictions and scores come from recorded inference. See [gallery building](gallery-building.md) to reproduce an export.

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
