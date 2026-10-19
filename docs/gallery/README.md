# ocracle gallery

This is a small static GitHub Pages site. It displays fixed, manually transcribed
card-line crops and **recorded output from the actual OCR CLI**. Nothing in the
browser invents a prediction or confidence score.

The current gallery and checkpoints cover English only. Every fixture records
its print language (`en`), and the language filter is populated from the manifest
so additional languages can be added. Recognizing every printed MTG language
remains part of the project objective; other scripts are not yet trained or
verified.

The images in `assets/` are real scan crops with a paired blur stress condition.
The nine reserved card names and exact print IDs in `assets/fixtures.json` must
stay out of custom training, including other printings of those same card names.
This prevents a model from learning the gallery's reference strings during MTG
fine-tuning. A pretrained general OCR model's original training corpus may be
unknown; its results are described as a baseline rather than a strictly
independent benchmark.

To rebuild fixtures (Python 3 and Pillow):

```sh
python3 scripts/gallery_fixtures.py
```

The script caches full scans under `.cache/gallery/`; only the small text crops
and provenance are shipped. It sends a descriptive User-Agent and spaces out
Scryfall requests. Original PNG dimensions are checked before applying fixed crop
coordinates. Printed text was manually checked against those images, rather than
taken from current Oracle text.

After building the Rust CLI and downloading its recognition model, record output:

```sh
python3 scripts/gallery_export.py \
  --recognition-model models/text-recognition.rten
```

`--executable`, `--passes`, `--detection-model`, `--model-config`, `--fixtures`, and `--output` can be
specified. These single-line crops use `--line`; text detection is not part of this
gallery run. Each record contains the complete inference JSON, the image hash,
and a character edit distance against the print transcription. The export also
hashes the model, executable and fixture manifest to identify the run.

Verify all checked-in observation hashes, model identities and metrics without
loading a model:

```sh
python3 scripts/gallery_export.py --check
```

Serve locally from the repository root:

```sh
python3 -m http.server 8000 --directory docs/gallery
```

Open <http://localhost:8000>. GitHub Pages can publish this directory as a static
artifact. Keep `index.html`, `style.css`, `gallery.js` and `assets/` together.

Confidence is uncalibrated: it is the geometric mean of emitted CTC character
posteriors, not the probability that the complete line is correct. Selecting
among preprocessing candidates can inflate that score. Character error rate
counts case and punctuation, with NFC normalization and collapsed whitespace;
values over 100% are possible if OCR inserts many characters. The small gallery
shows failure cases as well as successes and does not establish complete font,
set, language or special-treatment coverage.

Card image/text rights remain with Wizards of the Coast. Scryfall supplies the
scans and print metadata. See the per-crop provenance and the Wizards typography
sources linked on the page.
