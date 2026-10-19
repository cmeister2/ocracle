# Reviewed Latin and best-effort script fixtures

Thirteen real printed crops have clean and severely blurred versions. Latin
is a primary target; Hebrew, Arabic, Sanskrit and Ancient Greek are best effort.
These fixed examples expose failures without training on the evaluation cards.
They are a small diagnostic set, not evidence of complete language coverage.

| Language | Actual printing | Fields |
| --- | --- | --- |
| Hebrew | Glory, Judgment prerelease | Title, type, flight keyword |
| Latin | Raging Kavu, Invasion prerelease | Title, type, italic reminder line |
| Ancient Greek | Cleopatra, Exiled Pharaoh, Assassin's Creed | Title, type with polytonic marks |
| Arabic | Stone-Tongue Basilisk, Odyssey prerelease | Title, type |
| Sanskrit | Fungal Shambler, Apocalypse prerelease | Title, type, keyword |

[cards.json](../../training/rare-scripts/cards.json) retains the localized API records, scan URLs, source
digests and native dimensions. Each original scan was inspected to confirm that
it really shows the requested language. [annotations.json](../../training/rare-scripts/annotations.json)
records manually bounded crops and transcriptions. [review.png](../../training/rare-scripts/review.png)
shows the clean pixels. Boxes use original 672×936 coordinates without resizing.

Labels retain logical Unicode order, case, punctuation and combining marks.
Evaluation uses NFC and collapsed whitespace. Three explicit transcription
decisions correct limitations of the catalogue metadata:

- Hebrew `FB20` wide ayin in the flight keyword becomes ordinary logical ayin
  `U+05E2`; this is a declared presentation-form mapping, not mark stripping.
- The Latin type line uses its visible short hyphen. The italic reminder line
  keeps its full opening word, closing period and parenthesis.
- The Sanskrit type line retains the final visible visarga `ः`, omitted by the
  API's localized type field.

All five canonical card identities and Oracle IDs are excluded from project
training and checkpoint selection across languages. This explicit reservation
overrides their nominal training hash bucket. Generic upstream OCR pretraining
membership is unknown. English Oracle text never replaces a localized label.

```sh
python training/rare-scripts/prepare.py --check
python training/rare-scripts/prepare.py --check --download-sources
# Regenerate from verified cached or downloaded originals:
python training/rare-scripts/prepare.py --download-sources
```

The offline check verifies fixture digests, dimensions, reviewed labels, unique
observations, frozen provenance and training identity isolation. The source
check additionally verifies original scan hashes and byte-identical derivatives.
Damage is Gaussian blur σ2.4 native pixels followed by 30% bilinear downsampling.
Expected text stays intact even when an accent becomes unreadable.

[Paddle probes](../experiments/paddle.md) retain actual model outputs and
glyph/layout limitations. These models are not made comprehensive by adding a
profile entry. Wizards retains card imagery/text rights; these fixtures are not
relicensed under the Rust crate's MIT license.
