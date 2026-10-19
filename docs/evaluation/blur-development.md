# Fresh printed-line blur development corpus

This development cohort contains 64 genuine printed lines from 16 fresh canonical card identities in English, French, German, Spanish, Italian and Portuguese. Each line has seven predefined image conditions, for 448 observations. It targets ordinary flavor/reminder italics, historic print texture, and Latin accent detail. Sources, crop boundaries and literal labels were reviewed and frozen before any recognizer ran on these examples.

This is **development data**: it may inform model selection. It is not an untouched test set or a training corpus. Upstream pretrained model membership is unknown.

## Coverage and selection

| Language | Cards | Clean lines |
| --- | ---: | ---: |
| English | 4 | 16 |
| German | 3 | 12 |
| French | 3 | 12 |
| Spanish | 2 | 8 |
| Italian | 2 | 8 |
| Portuguese | 2 | 8 |

There are 16 title lines, 12 regular rules lines, 31 flavor lines and 5 italic reminder lines. Real Fifth Edition, Seventh Edition and Eighth Edition prints provide historic title/body texture and physical italic wording; Dominaria and Core Set 2021 provide modern treatment and localized accents. The accepted historic editions are English. This small cohort does not establish complete font, card, language or accent coverage.

The acquisition process checked canonical English names and Oracle IDs against existing English, Latin V1, Latin V2, punctuation, accent, rare-script, poster-lettering, Japanese-probe, color-development and gallery metadata. The frozen snapshot excludes 490 canonical names and 478 Oracle IDs across eleven metadata files. The verifier also checks the current versions of those files, so later ingestion of one of these identities fails the disjointness check. All printings and translations of a canonical identity are treated as the same identity.

Candidate discovery used the pinned Scryfall bulk snapshot recorded in [provenance.json](../../training/blur-corpus/provenance.json). Metadata helped locate scans and possible accents/reminders. It did not supply authoritative labels. The final selection used visible source language, legible physical text, historic/modern balance, and a fixed four complete lines per accepted card. No OCR score, prediction or model output was used to select sources, transcribe text or choose image conditions.

## Physical source review

[cards.json](../../training/blur-corpus/cards.json) records print ID, Oracle ID, canonical name, language, set, collector number, artist, source URL, original image-byte digest, dimensions and decoded RGB digest. English sources use Scryfall PNGs; localized sources use official Wizards Gatherer images identified by the localized multiverse ID. Missing source downloads observe a 150 ms delay and a descriptive user agent. Every fetched source must match its frozen hash.

Localized metadata does not guarantee localized pixels. Three older French/German candidates returned visibly English scans, including from their localized Gatherer URLs. They were rejected before annotation and are preserved in [rejected-sources.json](../../training/blur-corpus/rejected-sources.json) and [source-rejections.png](../../training/blur-corpus/source-rejections.png). Their pixels never enter the accepted corpus.

Review evidence is retained alongside the corpus:

- [Source contact sheet](../../training/blur-corpus/review/sources.png): all sixteen accepted physical prints.
- Clean line sheets [1](../../training/blur-corpus/review/clean-01.png), [2](../../training/blur-corpus/review/clean-02.png), [3](../../training/blur-corpus/review/clean-03.png), [4](../../training/blur-corpus/review/clean-04.png), [5](../../training/blur-corpus/review/clean-05.png), [6](../../training/blur-corpus/review/clean-06.png), [7](../../training/blur-corpus/review/clean-07.png), [8](../../training/blur-corpus/review/clean-08.png): native crops next to manually entered labels and bounds.
- [Degradation contact sheet](../../training/blur-corpus/review/degradations.png): a real italic line under all seven conditions, shown at native output size.

[annotations.json](../../training/blur-corpus/annotations.json) preserves complete physical line boundaries, case, diacritics, curly apostrophes/quotes, guillemets and punctuation. Labels are NFC. A crop can start or end mid-sentence because the physical line is the annotation unit. No mana/tap symbols are relabeled as ordinary text. The printed numeric minus in `Le Cyclope déchaîné gagne -2/-0` is transcribed as ASCII U+002D by explicit convention; the glyph pixels cannot determine a Unicode codepoint.

A second reviewer independently confirmed all 64 labels on the eight clean sheets. That reviewer also inspected the native German Dread Shade flavor and confirmed the exact line `„Der Wald, der einst die Vess-Ländereien`, including the low opening quote U+201E and `ä`. This low quote is absent from the frozen Paddle v6 small dictionary; it remains in the literal ground truth. [vocabulary-audit.json](../../training/blur-corpus/vocabulary-audit.json) records a set-membership audit against exact configuration hashes, with no inference. Do not normalize away a missing character to improve a score.

## Fixed image conditions

All transformations begin from the same native RGB crop; only the severe condition composes operations. No random draw, synthetic text rendering, super-resolution or model-driven enhancement is involved.

| Condition | Exact operation |
| --- | --- |
| `clean` | Native RGB pixels, PNG. |
| `gaussian-1p2` | Pillow Gaussian blur, σ = 1.2 native pixels. |
| `gaussian-2p4` | Pillow Gaussian blur, σ = 2.4 native pixels. |
| `downsample-50pct` | Bilinear resize to 50% of native width/height; no upscale. |
| `motion-9px` | Horizontal 9-pixel box convolution; replicate edges; integer half-up rounding. |
| `jpeg-q25` | JPEG quality 25, 4:2:0 subsampling (`2`), no optimize or progressive encoding. |
| `severe` | Gaussian σ = 2.4, then bilinear resize to 30%; no upscale. |

Resize dimensions use explicit half-up rounding. Gaussian and motion lengths are in native source pixels, so their effect varies with the original text size; native dimensions remain in the manifest. These operations probe defined degradation rather than claiming to represent every camera or scanner failure.

## Reproduction and schema

Use the pinned Pillow version in [training/requirements.txt](../../training/requirements.txt). The checked-in render environment is Pillow 13.0.0, JPEG library 6.2, zlib 1.3 and WebP 1.6.0. Source caches are ignored. Preserved image bytes can be checked without network access:

```sh
python3 training/blur_prepare.py --check
```

To acquire missing exact sources and reproduce all pixels/manifest records:

```sh
python3 training/blur_prepare.py --check --download-sources
python3 training/blur_prepare.py --rebuild --download-sources
python3 training/blur_prepare.py --check
```

The downloader accepts only the original frozen source digest. Changed imagery fails closed. Rebuilding requires the recorded rendering-library versions, reproduces the frozen manifest digest, and creates review sheets from the same annotations. The script imports no recognizer and runs no model. A local cache with a mismatching source or rendering environment fails verification rather than silently skipping reproduction.

[manifest.jsonl](../../training/blur-corpus/manifest.jsonl) has one record per base line/condition. Fields include `base_id`, relative `path`, literal `text`, `language`, canonical `card_name`/`oracle_id`, `card_id`, set/frame/collector number, `style`, `italic`, `split: development`, native exclusive-right/bottom `bbox`, exact `transform`, source URL/digests, native clean-pixel digest, image-byte digest, decoded image-pixel digest and output dimensions. Pixel digests hash row-major interleaved `RGB.tobytes()`; dimensions are recorded separately.

Keep every predeclared model/case result when comparing candidates. Report exact matches and character errors by condition/style/language, retain raw predictions and confidence definitions, and separate vocabulary impossibility from visual recognition failures. No full-card detection/layout claim follows from this provided-line evaluation.

Magic: The Gathering card imagery, printed text and artwork belong to Wizards of the Coast and the respective artists. Small research/evaluation crops retain exact source attribution; no ownership or endorsement is claimed.
