# ocracle

Local neural OCR for **Magic: The Gathering** card text, historic typography and
italics, with blur training experiments and a [simple live gallery](https://cmeister2.github.io/ocracle/).

Experimental: Latin-script languages take priority; non-Latin alphabets are best
effort. Phyrexian is a stretch goal. Coverage gaps and failures are documented.

Requires **Rust 1.94+** and Python 3 for the initial model download. OCR runs offline.

```sh
git clone https://github.com/cmeister2/ocracle.git
cd ocracle
python3 training/download_models.py
cargo run --release -- recognize card.jpg --language en
```

Choose one language per card: `en`, `es`, `fr`, `de`, `it`, `pt`, or `la`.
Use `--line` for a text crop. Output includes raw text, coordinates, method and an
uncalibrated confidence score. No Oracle lookup replaces the observed text.

- [Usage, library API and development](docs/guide.md)
- [Fonts](docs/font-coverage.md) and [languages](docs/language-coverage.md)
- [Training and evaluation](docs/training/english-crnn.md)
- [Model provenance and licenses](docs/models.md)

Changes use pull requests with the required **Success** CI check; stacked PRs are
supported. Code is MIT licensed; models and card imagery retain separate rights.
