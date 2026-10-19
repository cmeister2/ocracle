# Complete poster-lettering challenge

With each complete field supplied as a single line, the current Paddle profile
reads **6/10** title fields exactly. It reads
Nature's Lore, Gamble and Preordain in both clean and severe variants, but fails
the strongly arched Wrath of God and Decree of Pain fields. The MTG English CRNN
and generic Ocrs CRNN both score **0/10**. These are actual Rust CLI results,
including all original/enhanced candidates and their confidence scores.

The separate detected-text path scores **4/10**, and fixes neither arched title.
It can add artwork/neighbouring-field false positives; confidence selection makes
its character error rate worse than the original detector candidate.

This small frozen challenge exposes gaps in the goal of handling every historic
MTG lettering treatment. It does not establish complete coverage or isolate
font shape from title layout and segmentation.

## Sources and literal labels

Wizards' [2021 announcement](https://magic.wizards.com/en/news/announcements/now-entering-dr-lairs-secretorium-superdrop-2021-04-22)
lists all five hand-drawn cards in the concert-poster drop *Our Show Is on
Friday, Can You Make It?*. All five were included before inference, with no
selection based on OCR success. They are separate artists' custom lettering,
not five verified commercial font families.

| Printed title transcription | Print | Artist | Observed treatment |
| --- | --- | --- | --- |
| `NATURE'S LORE` | SLD189 | Jeff Soto | Ragged brush capitals and long flourishes |
| `WRATH·OF·GOD` | SLD185 | Jermaine Rogers | Psychedelic letters stretched around a face |
| `GAMBLE` | SLD188 | Sam McKenzie | Angular capitals and deep offset shadows |
| `DECREE of PAIN` | SLD187 | Ian Jepson | Arched block capitals around smaller lowercase `of` |
| `PREORDAIN` | SLD186 | Alexis Ziritt | Rounded poster capitals over a geometric background |

The labels come from manual inspection of the actual source pixels. They retain
display case and visible punctuation. Nature's painted apostrophe is represented
by U+0027; its brush contour cannot identify a digital font encoding. Wrath's two
visible round separators are represented by U+00B7 middle dots, rather than
replaced with canonical card-name spaces. The CRNN alphabets cannot emit that
middle dot, a separate structural limitation recorded for those observations.

[cards.json](../../training/style-eval/cards.json) stores fixed Scryfall print
IDs, complete localized metadata, source URLs and scan SHA-256 digests. All five
are marked high-resolution scans. [annotations.json](../../training/style-eval/annotations.json)
stores manual title labels, exact bounds, punctuation conventions and incidental
content. [review.png](../../training/style-eval/review.png) shows every clean
field and an enlarged severe counterpart; the actual inference inputs remain
the native crops recorded in [manifest.jsonl](../../training/style-eval/manifest.jsonl).

## Geometry and degradation

Each crop is one axis-aligned rectangle containing the complete lettering,
including its curved baseline, unusually tall letters, shadows and flourishes.
There is no rectification, flattening, masking, synthetic replacement or
character rearrangement. RGB pixels retain their original appearance.

Rectangles also retain incidental source content: Nature's adjacent Sorcery
lettering and part of its mana icon, Wrath's mana cost and central figure, and
small mana-symbol fragments around Gamble and Decree. These are explicitly
annotated. The expected output is the title, so copying a neighbouring field
would count as an extraction error. This is a complete title-field task with
layout complications, rather than a purified single-font test.

Every source has exactly two variants: an unchanged crop, and Gaussian blur
with sigma2.4 in source pixels followed by 30% bilinear downsampling in each
dimension. Pillow13.0.0 is pinned for exact PNG reproduction. The contact sheet
enlarges severe crops only for inspection; OCR receives their native dimensions.

## Actual model results

The benchmark uses the production `en` Paddle profile, retained `en-mtg` profile,
and pinned generic Ocrs base weights with an explicit grayscale CRNN
configuration. Model/configuration/registry files and the executable are copied
to an isolated snapshot before inference, so concurrent builds or training
cannot change the evaluated model. The offline model-integrity verifier runs
before the snapshot is used.

The CLI receives `--line --passes all` for every case/model. It bypasses text
detection and retains both preprocessing candidates. The selected result is the
engine's existing confidence choice; expected text never selects a pass. There
are 30 successful CLI runs and 60 retained candidate observations, with no
inference errors or omitted cases.

| Model | Original exact / CER | Enhanced exact / CER | Confidence-selected exact / CER |
| --- | --- | --- | --- |
| Paddle PP-OCRv6 small (`en`) | 6/10 / 25.93% | 6/10 / 29.63% | 6/10 / 25.93% |
| MTG English CRNN (`en-mtg`) | 0/10 / 54.63% | 0/10 / 50.93% | 0/10 / 53.70% |
| Generic Ocrs CRNN | 0/10 / 49.07% | 0/10 / 49.07% | 0/10 / 48.15% |

Paddle's selected result is exact on three of five clean fields and the same
three severe fields. Its clean Wrath prediction is `ARMGOt` (score0.461), and
clean Decree is `DECREPAIN` (score0.786). The complete title labels, malformed
predictions, per-pass scores, source hashes and actual inference JSON are retained
in [results.json](../../training/style-eval/results.json). Enhancement changes
some erroneous outputs without fixing either arched title.

Scoring uses NFC and collapsed whitespace, preserving case, apostrophes and
middle dots. CER is total character edits divided by108 reference characters
across the ten observations; it may exceed100% for insertion-heavy output. A
case-folded exact check is a separately labelled diagnostic and yields no extra
exact matches in this run. Confidence is an uncalibrated model posterior score,
not a probability that the title is correct.

## Detected-text follow-up

The same ten frozen crops were then passed to the actual detector path using
`--language en --layout text --passes all`, without `--line`. The detector and
recognizer were snapshotted and pinned. No crops, expected labels, model weights
or case inclusion changed; the original three-model report remains intact.
All detection boxes, line outputs, candidate text and scores are preserved in
[results-detected.json](../../training/style-eval/results-detected.json).

| Paddle detected-text path | Exact | CER |
| --- | ---: | ---: |
| Original candidate | 4/10 | 60.19% |
| Enhanced candidate | 0/10 | 139.81% |
| Confidence-selected output | 4/10 | 107.41% |

Gamble and Preordain remain exact in both severities. Nature's title is split or
combined with adjacent Sorcery/artwork output, and neither Wrath nor Decree is
recovered. For clean Wrath, confidence selects the enhanced candidate containing
many false lines over the shorter original candidate; the insertion errors
explain CER above100%. These are ordinary title-field detector results, not
full-card detection tests. They show that switching from forced-line compression
to the current detector does not solve this challenge. No engine changes or
reference-based candidate selection were made after seeing the results.

## Isolation and interpretation

[provenance.json](../../training/style-eval/provenance.json) pins the source
selection, annotations, manifest, review sheet and original corpus-exclusion
audit. No canonical name, Oracle ID or print ID from these five cards appears
in the current English, Latin V1 or Latin V2 corpus selections or available
materialized manifests. Future Latin acquisition also reserves their identities.
No model was trained or selected using these observations. Upstream pretrained
model membership remains unknown.

There are only five card identities and one clean source per treatment. The
results mix recognition, punctuation coverage and two-dimensional title layout;
they do not prove that a different segmentation or rectification system could
not read these prints. The concrete remaining gaps are the intact arched Wrath
and Decree title fields, even before additional degradation. New training and
validation should use independent lettering sources while preserving this set
as a frozen challenge.

## Reproduction

From the repository root, fetch the pinned inference weights and build the CLI:

```sh
python3 training/download_models.py
cargo build --release
python3 -m venv /tmp/ocracle-style-venv
/tmp/ocracle-style-venv/bin/pip install Pillow==13.0.0
/tmp/ocracle-style-venv/bin/python scripts/verify_models.py --require-models
/tmp/ocracle-style-venv/bin/python training/style-eval/prepare.py --check --download-sources
/tmp/ocracle-style-venv/bin/python training/style-eval/score.py --check
/tmp/ocracle-style-venv/bin/python training/style-eval/score.py --detected --check
/tmp/ocracle-style-venv/bin/python training/style-eval/score.py --output /tmp/ocracle-style-results.json
/tmp/ocracle-style-venv/bin/python training/style-eval/score.py --detected --output /tmp/ocracle-style-detected-results.json
```

`prepare.py --check` verifies all ten committed PNGs, the frozen annotations and
manifest, and current training-identity exclusion without rewriting artifacts.
If original scans are available, it reproduces the exact crop/degradation bytes.
`--download-sources --source-directory /tmp/ocracle-style-sources` performs an
independent replay into a separate scan cache. All downloads fail closed on
source digest mismatch. Missing source scans do not prevent verification of the
already committed crop hashes; they prevent claiming a fresh source replay.
`score.py --check` verifies complete case/model membership, raw inference JSON,
recognizer/detector identities, confidence selection and recomputed per-case/
aggregate metrics without running inference or rewriting either result file.
