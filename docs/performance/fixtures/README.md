# Performance fixtures

`manifest.json` fixes 50 gallery line crops, three multiline body crops, and three
whole-card scans. The three complete scans are already pinned by URL and SHA-256
in the English gallery manifest. They are cached under `.cache/gallery/`; this
directory ships only their smaller body crops. Restore missing scans explicitly:

```sh
python3 scripts/performance_suite.py --prepare --download-sources
```

The command verifies source hashes before producing derivatives. It never selects
images using OCR output. Full-card title, type and body transcripts were visually
reviewed on 2026-10-20. Historic printed wording, punctuation and diacritics remain
literal. Source metadata and crop bounds are retained in the manifest.

Line and multiline CER use the complete input crop. Whole-card CER is **partial**:
each predicted line is assigned to a fixed title/type/body rectangle when its
center lies inside that rectangle. Predictions retain their actual reading order.
Missing text counts as deletions; extra text within a scored rectangle counts as
insertions. Mana symbols, artwork and collector information are not transcribed
or included in CER, but their detections and recognition still contribute to
latency and remain visible in the overlays. This is not a detector recall or
whole-card transcription benchmark. NFC normalization and collapsed whitespace
are the only text normalization.

Run every configuration sequentially on the same host and fixture selection:

```sh
python3 scripts/performance_suite.py \
  --model models/ppocr-v6-medium.onnx --config models/paddle-v6-medium/en.json \
  --label "Medium baseline" --layout standard_card --threads 4 --iterations 20 \
  --workload card --workload multiline --output /tmp/before.json
```

`--detector PATH` selects a detector artifact; `--executable PATH` selects a
preserved baseline binary. Both rely on the benchmark harness's explicit
`OCRACLE_DETECTOR_MODEL` and `OCRACLE_LAYOUT` environment controls. `--workload line`
selects 30 English gallery cases. `--language` selects localized crops, requiring
a configuration that declares that language. Repeated `--sample ID` flags select
particular fixtures. `--layout` applies only to whole-card cases; crops always use
`text` to avoid applying a card artwork mask to actual text. Every run records
the effective layout plus exact binary, model, configuration,
detector and image hashes, raw timing samples, predictions, line boxes and scores.
Interrupted runs retain their completed observations and a non-complete status.

Generate comparison artifacts with a Python environment containing Pillow and
matplotlib:

```sh
python3 scripts/performance_report.py \
  --run /tmp/before.json --run /tmp/after.json \
  --output docs/performance/results
```

The report rejects unequal fixture sets or changed references, recomputes CER,
checks timing summaries against raw samples, and checks all boxes against original
input dimensions. The latency chart uses the geometric mean of each case's warm
median, giving cases equal multiplicative weight. Individual medians/p95 values,
speedups and error regressions remain in `summary.json`. Chart target lines show
the 66.7 ms budget; desktop measurements do not establish Android FPS. An input
image is resized only for displaying the original-coordinate OCR boxes.

These are clean fixed scans and deliberately degraded line crops, not camera
sequences. They do not cover motion, glare, phone processors, sustained thermal
load or every printed language/layout. Identities are excluded from custom
training; membership in upstream model training is unknown. Card imagery and
text retain Wizards of the Coast's rights, with scans supplied by Scryfall.
