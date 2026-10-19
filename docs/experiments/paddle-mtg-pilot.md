# Real MTG adaptation pilot

This is a new experiment using the native Paddle path. The original feasibility
scripts and `parity.json`, `smoke.json`, `export.json` remain unchanged.
No production model/profile is modified. The completed 100-update pilot selected
step 25 using the predeclared validation rule. A single later assessment improved
modestly over the official small recognizer on the fresh development cohort;
motion blur and some individual languages still need work.

## Prepared inputs

`mtg_prepare.py` creates `mtg-dataset.jsonl` and `mtg-data.json`, with image
outputs under ignored `runs/mtg-data/`. It replays pinned scan bytes and reviewed
boxes into RGB rather than inheriting the CRNN's grayscale pixels:

| Role | Lines | Source/label policy |
| --- | ---: | --- |
| Supervised physical training | 81 | 58 accepted V2 train titles, 20 accepted body lines, 3 independently accepted punctuation lines; 61 canonical identities |
| Fixed V2 validation | 23 | Accepted validation-title crops across 21 canonical identities, original canonical split retained |
| English teacher anchors | 179 | Real train scans; old normalized labels omitted entirely |
| Synthetic practice | 348 | Exact training labels and procedural glyph lines in four pinned regular/italic open fonts |

The body metadata contains21 accepted records; canonical Opt is reserved and
excluded, leaving the same20 lines used in frozen V2. The three punctuation
crops add literal physical `¡` and `„` supervision. Synthetic `ẞ` practice is
explicitly weaker evidence than a physical print. Every synthetic character
must occur in the selected font's cmap; missing glyphs never become labeled
replacement boxes. Source font hashes/licenses are recorded.

Canonical card names and Oracle IDs keep train/validation roles together
across languages and printings. Existing gallery, Latin/style/rare tests and
the fresh blur corpus are reserved from training. The preparer does not run
OCR or read existing test labels for selection. Preparation verifies the old
reviewed grayscale crop hashes before producing RGB crops; punctuation RGB
bytes must reproduce exactly. Teacher anchors carry no `text` field.

`--check` verifies the prepared manifest, all pixel bytes, input annotation and
font hashes, preparation implementation, canonical splits and newly reserved
identities. A new source annotation or conflicting holdout stops training.

## Reproduce the bounded benchmark

Use the isolated CPU environment and source bootstrap described in
[the feasibility documentation](paddle-training.md).
Run training in a separate working copy: the historical trainer writes its
report paths and must not replace this frozen evidence during an audit.
Source downloads are cached by exact digest; missing scans are fetched from
their pinned provenance URLs. Only the new optional font coverage dependency
is required beyond that environment:

```sh
/tmp/ocracle-paddle-training-venv/bin/python -m pip install --require-hashes -r training/experimental_paddle_training/requirements-mtg.lock
/tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/mtg_prepare.py
/tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/mtg_prepare.py --check
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python -m unittest discover -s training/experimental_paddle_training -p 'test_*.py'
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/mtg_train.py --mode benchmark
```

The fixed seed is 20261021. The benchmark starts again from the official native
checkpoint, appends `¡`, `„`, `ẞ` without moving any old index, and performs
exactly five updates. They include all three real punctuation examples, one
body and one title, with Gaussian/downsample/JPEG/motion/combined conditions.
Each update also sees an exact synthetic line and a clean real English anchor.
Augmentation parameters, tensor hashes, selected row IDs, losses, gradients
and timings are recorded in `mtg-benchmark.json`.

All modules stay in evaluation mode for dynamic-width inference pooling,
fixed BN statistics and disabled dropout. Raw-logit official CTC uses full
sequence lengths. The objective is physical CTC +0.25 synthetic CTC +1.0
teacher KL, with full expanded student probabilities. Adam rates are 1e-4
for the classifier, 1e-5 for LightSVTR and 5e-6 for the upper LCNet block; global
gradient clipping is four. This is a small adaptation recipe, not a reproduction
of upstream auxiliary NRTR training.

Eight fixed clean training anchors measure decoded-text preservation before
and after, without interpreting the normalized legacy label as gold. This
is not an independent clean accuracy evaluation.

## The gated 100-update pilot

The fresh blur corpus remains reserved from training and is not used for this
checkpoint choice, even after its labels freeze. It will support a later model
family/default comparison. The pilot selects only on the 23 already-frozen V2
validation-title crops across 21 canonical identities and the fixed English
teacher-preservation term.

After the owner confirms readiness, the explicit command is:

```sh
OMP_NUM_THREADS=1 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/mtg_train.py --mode pilot --pilot-confirmed
```

The pilot requires verified source/data bindings and explicit confirmation.
It performs 100 updates, evaluating the 23 V2 titles under four fixed conditions:
clean, Gaussian sigma 2.4 source pixels, 30% bilinear downsample/restore, and
JPEG quality 35 with subsampling 2. Thus each checkpoint has 92 observations.
Evaluation/save steps are 0/25/50/75/100. The initial expanded source checkpoint
is eligible, so training selects step 0 if every updated checkpoint is worse.

The score, declared before baseline evaluation, is
`0.90 × character-weighted CER over the 92 variants + 0.10 × decoded-text
disagreement rate on the eight fixed clean English training anchors`.
Lowest wins; ties retain the earliest checkpoint. The original unexpanded
teacher is also scored for a baseline. Every scheduled native checkpoint is
saved, with its hash and validation result. No existing frozen test result or
fresh blur-corpus result participates in this choice.

The five-update timing projects roughly 11–12 minutes for 100 CPU updates before
baseline/selection evaluation and initialization overhead. Allow 15–20 minutes
for the complete pilot on the measured CPU, then separately prove adapted
export/runtime parity and evaluate independent frozen tests after selection.
The owner explicitly authorized this bounded pilot after reviewing the timing;
this document does not authorize additional or longer runs.

## Frozen pilot and export evidence

The pilot completed once. All 116 BN buffers stayed unchanged and the
classifier, LightSVTR and upper LCNet block received nonzero updates. Selection
used the following recorded results; every row preserves all 92 predictions:

| Step | Exact /92 | Character edits /1,824 | Clean anchors preserved /8 | Selection score |
| --- | ---: | ---: | ---: | ---: |
| 0 | 79 | 13 | 8 | 0.00641447 |
| **25** | **80** | **12** | **8** | **0.00592105** |
| 50 | 80 | 13 | 8 | 0.00641447 |
| 75 | 80 | 14 | 8 | 0.00690789 |
| 100 | 79 | 14 | 8 | 0.00690789 |

This is a one-character validation improvement, not independent proof of a
general gain. The recorded loop took 1,041.5 seconds including four scheduled
validation checks. Baseline evaluation and initialization add wall time.
The original process was PID219769, started
2026-10-19T18:37:42.521505+00:00; it has ended. The complete history is in
[mtg-pilot.json](../../training/experimental_paddle_training/mtg-pilot.json).
Selected native SHA256 is
`451b0e2ada65195045037b3d1db2ad60403e41599691e093415dfa3c2c1625a4`.
Native checkpoints and exported model files remain ignored local outputs, not
distributed or registered profiles. Reproduction requires the source/data replay
and the bounded pilot above; training weight bytes are not promised identical
across machines or dependency versions.
The frozen checker/export proof accepts the archived checkpoint and report
hashes. A newly trained replica has different process/timing evidence and must
retain its own report rather than substituting it into this archive. If its
weights differ, it needs a separately bound export assessment. The historical
trainer remains unchanged so its original report/code bindings stay valid.

[mtg_export.py](../../training/experimental_paddle_training/mtg_export.py) uses
the pinned official Paddle2ONNX path without changing the earlier smoke proof:

```sh
OMP_NUM_THREADS=2 /tmp/ocracle-paddle-training-venv/bin/python training/experimental_paddle_training/mtg_export.py --runner /tmp/ocracle-paddle-target/release/paddle-rten-probe --cli /tmp/ocracle-before-colour-same-compiler
```

The selected export is 21,331,267 bytes, SHA256
`f551ae37ba046b507a53f2c6b3a1f82077056815e68be55ebb9f7f78157cb09e`.
[mtg-export.json](../../training/experimental_paddle_training/mtg-export.json)
records five native/ONNX Runtime probes, four ONNX Runtime/RTen probes and two
actual Rust CLI probes. Maximum probability differences are 3.38e-5 and
8.82e-6 respectively; all argmax sequences agree. Batch and width remain
dynamic; all 18,713 output classes retain their indices, including the three
appended classes. The procedural new-glyph probe still misses some added glyphs;
graph parity is not a punctuation-accuracy claim.

## One later assessment, without retuning

After selection and export, the adapted checkpoint was evaluated once on all
448 frozen fresh development inputs, using the exact small-baseline CLI binary
and the production original/enhanced confidence-selection policy recorded at
that evaluation. The current medium-original default was selected separately.
This cohort contains 64
manually transcribed lines from 16 fresh identities, seven conditions per line,
including 36 italic base lines. It was excluded from training and checkpoint
selection. Every raw process output, both candidates, source/config/image/model
hash and literal label is retained in
[mtg-development.json](../../training/experimental_paddle_training/mtg-development.json).

| Development subset | Small exact | Adapted exact | Small CER | Adapted CER |
| --- | ---: | ---: | ---: | ---: |
| All 448 | 313 | 324 | 4.551% | 4.294% |
| Clean 64 | 59 | 61 | 0.231% | 0.139% |
| Italic 252 | 154 | 160 | 5.622% | 5.381% |
| Motion blur 64 | 14 | 15 | 23.961% | 22.992% |
| Severe 64 | 29 | 30 | 4.986% | 4.709% |

CER counts Unicode character edits after NFC and collapsed whitespace, retaining
case, accents and punctuation. The overall denominator is 15,162 reference
characters; counts use the pass selected by that historical confidence policy.
Expected text did not select a pass. Portuguese CER increased from 5.408% to 5.561% across
56 observations despite the aggregate gain. Motion blur remains a substantial
failure class. This assessment did not trigger additional training or a new
checkpoint choice. A separate fresh comparison subsequently selected the
official medium model with its original pass as the default. This pilot only
establishes a trainable small-model path and its measured result.

The fresh cohort contains seven variants of one line beginning with the newly
added `„`; none of the adapted selected outputs emits that glyph, including the
clean case. It contains no `¡` or `ẞ` references. Thus the expanded head and
successful gradients do not yet demonstrate independent recognition of the
new classes. A later experiment needs separate training/development evidence
for punctuation, without repurposing these observed cases as training labels.

The separate original 50-case English/Latin assessment also ran once. Both the
adapted model and stored small baseline have 46/50 exact and 4/906 edits
(0.4415% CER), comprising English 27/30 and Latin 19/20; all 25 clean cases are
exact. Full results are in
[mtg-gallery.json](../../training/experimental_paddle_training/mtg-gallery.json).
The stored baseline uses a different executable hash, disclosed in the report;
it was not rerun. Eight English italic flags were corrected from frozen fixture
metadata after inference; the report records the correction and unchanged raw
stdout hashes. Predictions and aggregate metrics did not change.

These are adaptation-held-out identities; upstream pretraining print identities
are unknown. Neither corpus proves universal font, language or blur coverage.
The experimental model remains unregistered. This pilot did not change
gallery, default-model or UI files; subsequent changes to those files belong
to separate work.

## Offline verification and assessment replay

All evidence checks use only Python's standard library and perform no inference,
training or downloads:

```sh
python3 -S training/experimental_paddle_training/mtg_report_check.py
python3 -S training/experimental_paddle_training/mtg_export_check.py
python3 -S training/experimental_paddle_training/mtg_assess.py --cohort development --check
python3 -S training/experimental_paddle_training/mtg_assess.py --cohort gallery --check
```

The pilot checker verifies frozen source/data/report hashes, all 100 seeded
training choices and augmentation parameters, all five checkpoint inventories,
validation membership, recomputed edit totals and the predeclared score. It
allows absent ignored caches explicitly and checks present bytes;
`--require-caches` additionally requires all 631 pixel caches, six native
checkpoint entries and the official source checkpoint. It does not claim an
inference or optimizer replay. Corruption tests exercise wrong sample IDs,
augmentation parameters, targets, metrics, scores, tensor hashes, checkpoint
selection, nonfinite losses and unsuitable teacher labels.
Its default command also checks the selected export and both complete assessment
reports through the frozen evidence manifest. `--pilot-only` restricts the
command to training evidence. No ignored model or pixel cache is required by
the default check.

For a new assessment replay after reproducing the selected export, use
`mtg_assess.py --cohort development` and then `--cohort gallery`, with `--cli`
pointing to the executable built from the pinned source. The recorded
executable SHA is enforced to keep comparisons meaningful. A different build
must be declared as a new experiment, not silently substituted. Per-case raw
records allow interrupted runs to resume without repeating completed inference;
there is no automatic retry of failed processes. No assessment command performs
training, checkpoint selection or profile registration.
