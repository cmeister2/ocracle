# Shared Latin-accent experiment

This separate corpus contains 7,984 lines from 120 localized M21 prints: English,
Spanish, French, German, Italian, and Portuguese. It has 120 actual title crops
and 7,864 synthetic regular/italic lines. Every title box was visually reviewed
on the source scans. Canonical card identity stays in the same split across
all languages. Alternate frames and planeswalkers were excluded from this
initial standard-title cohort.

Localized labels use Scryfall `printed_name`, `printed_type_line`, and
`printed_text`; English Oracle fallback is permitted only for English synthetic
lines. NFC preserves accents, case, and punctuation. Typographic apostrophes on
scans follow the plaintext punctuation convention in Scryfall's printed fields.
Synthetic labels omit API brace spelling for mana symbols. The 129-character
alphabet covers the acquired labels; this is not a complete orthographic
inventory for all six languages.

The open URW regular/italic faces are checked against their character maps;
no characters in this frozen corpus were omitted for missing font glyphs.
Font files retain their upstream AGPL-3.0 license and font embedding exception.
Source card text and scans retain Wizards' rights. The repository stores scan
URLs/digests and boxes, rather than the source card images or font programs.

Reproduction uses the same Python environment as the English baseline:

```sh
python training/download_models.py --checkpoint
python training/latin_prepare.py
python training/latin_train.py --steps 600
rten-convert models/ocracle-mtg-latin.onnx models/ocracle-mtg-latin.rten
```

Materialization enforces frozen scan/font digests and the manifest digest.
[Independent replay](../../training/latin-reproduction-validation.json) reproduced all 7,984
PNGs byte for byte in a separate working directory.

The experiment starts from the official generic checkpoint, freezes its
convolutional/GRU weights and batch-normalization statistics, and trains the
Unicode classifier with balanced languages and optical damage. A penalty keeps
existing Latin classifier rows near their pretrained values. Checkpoints are
selected by fixed multilingual validation CER, with the test split used only
for before/after reporting. The [training report](../../training/latin-report.json) records
the selected run and per-language/source/damage metrics.

The English baseline weights and report are preserved. This experiment does
not establish support for every historical font, every glyph, or any of the
remaining printed scripts. The full [language inventory](../language-coverage.md)
continues to track those gaps.

The [independent printed-accent evaluation](../../training/latin-eval/README.md) exposed a
material limitation: only 2 of 20 clean/blurred titles were exact (6.67% character
error rate). Many missing accents received scores above 0.99. Two required
characters, `Ä` and `ï`, are absent from this alphabet; three other accented
classes have no training observations. The internal aggregate improvement
therefore does not establish reliable accented-language recognition. These
weights stay experimental and are absent from the ready-profile registry.
