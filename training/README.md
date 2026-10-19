# MTG recognition training

The training workflow below produces the experimental **English CRNN**
(`--language en-mtg`). The default Latin-language profiles use an upstream colour
model evaluated on MTG, with [pinned provenance and actual Rust comparisons](experimental_paddle/README.md).
Latin-script languages, historic fonts, italics and blur are the priority;
other alphabets are best effort. [Language coverage](../docs/language-coverage.md)
tracks the remaining gaps. Phyrexian is a separate stretch goal.

This trains the neural recognizer used by `ocracle`. The detector locates text;
the recognizer reads image pixels directly with convolutional visual features,
a two-layer bidirectional GRU, and CTC. Card names are labels for training and
scoring, never a database used to correct recognition results.

The frozen corpus has 6,902 lines from 126 acquired English prints, representing
120 card identities. It contains 6,672 synthetic lines and 230 actual scan crops,
including 74 italic scan lines. The nine sampled cohorts are Alpha, Seventh and
Eighth Edition, Time Spiral, Magic 2015, Core Set 2021, Future Sight's future
frame, Planar Chaos's colorshifted frame, and Amonkhet Invocations. These samples
cover several historic and special treatments; they do not establish support
for every Magic typeface. See [font coverage](../docs/font-coverage.md).

## Reproduce the distributed model

Use Python 3.12 and the open URW Base35 fonts. On Debian/Ubuntu:

```sh
sudo apt-get install fonts-urw-base35 python3-venv
python3 -m venv training/.venv
training/.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
training/.venv/bin/pip install -r training/requirements.txt
training/.venv/bin/python -m unittest discover -s training -p 'test_*.py'
training/.venv/bin/python training/reproduce.py
```

`reproduce.py` downloads digest-verified base weights, renders the frozen
synthetic labels, fetches digest-verified card scans, applies the frozen crop
annotations, fine-tunes for 360 updates, evaluates held-out lines, exports ONNX,
and converts it to `models/ocracle-mtg.rten`. CPU training takes several minutes.
`--skip-train` materializes and checks the data alone. `--eval-limit 0` scores
every held-out line instead of the bounded 96-line subset per split.

The reproduction path uses [frozen card metadata](corpus/cards.json) and
[reviewed crop annotations](corpus/real-lines.json). It does not repeat the
initial crop discovery, so model changes cannot silently change the labeled
corpus. Font hashes are verified against [provenance](corpus/provenance.json). The runner
also requires the exact frozen line count, manifest digest, and aggregate PNG
digest before starting training. An independent replay reproduced every one of
the 6,902 PNGs byte for byte; see [reproduction validation](reproduction-validation.json).
Source card scans and font files are not redistributed. A changed upstream scan
fails its digest check; use a matching cached original rather than silently
accepting a different image.

The training seed, optimizer, update count, loss history, exact sample counts,
and paired base/fine-tuned scores are recorded in [report.json](report.json).
Dependency versions and random seeds are pinned; exact floating-point weights
can still vary across hardware. The distributed weights have their own digest in
[the model manifest](../models/manifest.json).

## Training conditions and evaluation

Each training sample is selected with a 25% probability from the actual scan
crops, otherwise from the full training split. Training mixes clean text with
Gaussian blur, severe Gaussian blur (sigma 1.6–2.8 native pixels), horizontal
motion blur (3–7 pixels), reduction to 25–55% resolution followed by enlargement,
JPEG quality 18–45, and combined downsampling/blur/JPEG. Text colors and rendered
sizes also vary. The eight synthetic faces are regular and italic variants of
Nimbus Roman, P052, URW Bookman, and Nimbus Sans. They are open substitutes,
not copies of Goudy Medieval, Matrix, Beleren, or Magic's rules typefaces.

Splits use SHA-256 of the case-folded card name: bucket 0 is test, 1 validation,
and 2–9 training. Every printing, crop, font, and damaged variant of a card stays
in its split. Gallery card identities are excluded explicitly, including the
special treatments. Tests check both name and Oracle-identity isolation.

Each selected validation/test line is scored under seven fixed conditions:
clean, mild Gaussian, severe Gaussian, motion, downsampled, JPEG, and combined.
Scores include character error rate and exact-line accuracy by damage, style,
font/cohort, and real/synthetic source. Primary metrics trim only boundary
whitespace; a `raw-untrimmed` group preserves the original result. Evaluation
decodes every CRNN output step, with blank index 0 and repeated-character CTC
semantics matching Rust. Test scores are descriptive comparisons before and
after a fixed training schedule, never a checkpoint-selection signal.

The original six cohorts' real title/body crops were discovered by requiring
baseline OCR to match an exact metadata substring. They are biased toward
already readable text. Future Sight, Planar Chaos, and Invocation title crops
were manually bounded and included independently of baseline OCR success.
Scores on these selected crops do not establish population-wide accuracy.
Severe blur sometimes erases distinctions that no OCR model can recover.

To check ONNX tensor parity and actual Rust transcripts on held-out regular and
italic crops after exporting:

```sh
cargo build --release
training/.venv/bin/python training/check_export.py
```

This writes [export-validation.json](export-validation.json), including dynamic
width/batch tensor checks and the exact deployed model digest.

## Extend the corpus

For a new local experiment, place metadata in `training/data/cards.json`, then:

```sh
training/.venv/bin/python training/prepare.py --font-dir /path/to/licensed/fonts
training/.venv/bin/python training/bootstrap_scans.py
training/.venv/bin/python training/bootstrap_body.py
training/.venv/bin/python training/special_crops.py
training/.venv/bin/python training/train.py --steps 360 --eval-limit 96
training/.venv/bin/rten-convert models/ocracle-mtg.onnx models/ocracle-mtg.rten
```

`prepare.py` initially uses the frozen metadata snapshot; edit an experiment's
local metadata deliberately to add prints. It renders labels and overwrites the
working manifest. Bootstrap scripts then append actual title and body crops.
`special_crops.py` contains manually reviewed boxes for the frozen special
prints; new special titles need their own reviewed annotations. Body bootstrap
requires whole-word exact metadata substrings. Review accepted labels and boxes
before adding them to the frozen corpus; never use approximate Oracle matches
as printed ground truth. The frozen reproduction path does not need these
bootstrap scripts.

English quotes/dashes/accents are normalized to the pretrained Latin alphabet.
Synthetic rules omit brace-encoded mana/tap symbols rather than teaching the
network that a glyph is printed as its API spelling. Mana symbols, non-Latin
scripts, and every Secret Lair/showcase font remain outside verified coverage.
`--font-dir` accepts additional locally licensed `.ttf`/`.otf` files; keep their
license and provenance with a future corpus revision.

## Rights and attribution

Training code is under the repository's MIT license. The upstream neural
weights are Robert Knight's Ocrs models, licensed CC BY-SA 4.0 and trained on
HierText and synthetic data. The adapted weights retain that model license;
see [model attribution](../models/README.md).

Scryfall supplies card metadata and card-image URLs. Magic card text and scans
remain Wizards of the Coast content, used here for noncommercial fan OCR
research; they are not relicensed as MIT training code. URW Base35 fonts are
AGPL-3.0 with the upstream font embedding exception; the repository stores font
names and hashes rather than font programs. Sources:
[Scryfall API](https://scryfall.com/docs/api),
[Wizards Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy),
[URW Base35](https://github.com/ArtifexSoftware/urw-base35-fonts),
[Ocrs model training](https://github.com/robertknight/ocrs-models),
[HierText](https://github.com/google-research-datasets/hiertext).

## Unicode profile initialization

`network.py` supports distinct Unicode alphabets and retains the shared visual
backbone when initializing a language-specific classifier. The initializer
copies old character and blank rows by character identity. New glyph rows are
untrained until actual language training and held-out evaluation have completed.

```sh
training/.venv/bin/python training/language_head.py \
  --cards localized-cards.json --language fr \
  --checkpoint training/checkpoints/ocracle-mtg.pt \
  --output-dir training/runs/fr-initialized
```

Input card objects must all have the selected `lang` and localized printed
fields. The helper preserves NFC orthography; it refuses to substitute English
Oracle text for missing localized labels. Latin classes remain available for
artist and collector credits. A separate shared Latin-accent experiment uses
`latin_prepare.py` and `latin_train.py` for English, French, German, Italian,
Portuguese and Spanish without changing this English baseline. Other scripts,
RTL reading order and vertical layouts need their own data and validation.
