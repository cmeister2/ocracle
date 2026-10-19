# Building the gallery

The static site in `docs/gallery/` displays actual printed card crops and recorded
Rust OCR output. `assets/catalog.json` connects independent fixture bundles to
model runs. The page shows eight fixed examples using the small Paddle model,
with original and severely blurred versions visible together. Four flavor and
reminder italic examples appear first. There are no model, language or image
selectors. Complete JSON bundles preserve the broader English and localized
evaluations, including the frozen CRNN comparisons.

The original English fixture manifest, crop images and two original result files
are frozen. Further language examples live in separate bundles. The current
localized bundle contains ten manually checked accented titles from French,
German, Spanish, Italian and Portuguese printings. Their exact bytes and source
digests come from the independent test set in `training/latin-eval/`.

To rebuild image fixtures:

```sh
python3 scripts/gallery_fixtures.py
python3 scripts/gallery_latin_fixtures.py
```

The first command recreates the original English bundle and should only be run
when intentionally reviewing its fixed sources. Downloaded full card images are
cached; only small text crops are shipped. The second command imports the frozen
localized test crops without downloading or rerendering them.

After downloading models and building the CLI, record new Paddle observations:

```sh
python3 training/download_models.py
cargo build --release --locked
python3 scripts/gallery_export.py --profile-per-language \
  --output docs/gallery/assets/results-paddle.json
python3 scripts/gallery_export.py --profile-per-language \
  --fixtures docs/gallery/assets/latin/fixtures.json \
  --output docs/gallery/assets/latin/results-paddle.json
```

Every invocation reads actual pixels through the Rust CLI with `--line`. The crop
coordinates are human supplied; text detection is not measured by these runs.
The selected preprocessing method, all competing candidates, confidence, model
identity, executable hash and measured character errors are retained. Registered
profiles use the printing's language and record its exact configuration digest.
Reference strings never select predictions or replace OCR output.

Validate the entire catalog offline:

```sh
python3 scripts/gallery_export.py --check
```

This verifies all bundle variants, source and crop hashes, model registration,
configuration digests, print-language identity, retained candidates and metrics.
Local model files are checked if present; the Pages job needs no model download.
Original frozen observations remain traceable to their historical executable.

Serve the site locally:

```sh
python3 -m http.server 8000 --directory docs/gallery
```

Open <http://localhost:8000>. Pages publishes that directory as a static artifact.
No upload or remote OCR service runs in the browser.

Confidence is an uncalibrated geometric mean of emitted CTC character posteriors,
not the probability that a whole line is correct. Selecting among preprocessing
candidates can raise the score while worsening the text. CER uses Unicode
character edit distance after NFC normalization and collapsed whitespace; case,
punctuation and diacritics count. Severe blur deliberately removes detail, and
errors remain visible.

Custom fine-tuning excludes these fixed card identities. Membership in an
upstream pretrained model's original training data is unknown. These small
examples do not establish complete language, typeface, text-role or card-layout
coverage. Non-Latin scripts are best effort. Card image and text rights remain
with Wizards of the Coast; Scryfall supplies scans and printing metadata.
