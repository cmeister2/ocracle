# Independent printed-accent evaluation

Ten visually reviewed, real printed titles provide two cases each for French, German, Spanish, Portuguese and Italian. Each has a clean crop and a severe-blur variant. All canonical card names hash to the existing **test** split and are absent, by name and Oracle ID, from both frozen English and Latin corpora. These are evaluation cases, not training or checkpoint-selection data.

| Language | Printed titles |
| --- | --- |
| French | Sphère de capture; Renaissance fongoïde |
| German | Licht der Verheißung; Gargaroth-Ältester |
| Spanish | Acólita de Basri; Dríada Quirion |
| Portuguese | Falha na Conjuração; Névoa Pestilenta |
| Italian | Opportunità; Crudele Realtà |

[cards.json](cards.json) freezes print IDs, localized labels, manual transcriptions, source URLs and hashes, original-image crop coordinates, and review notes. [review.png](review.png) displays the reviewed clean crops. [manifest.jsonl](manifest.jsonl) has twenty records compatible with the line-corpus fields: `path`, `text`, `card_id`, `oracle_id`, `card_name`, `language`, `split`, `source`, `kind`, `font`, `italic`, plus `severity` and provenance. Paths are relative to this directory. English Oracle text never supplies the expected label.

`ï` and `Ä` are absent from the frozen 129-character Latin head. Both cases remain in the evaluation as explicit structural gaps; do not drop the accents or discard those samples. The other eight titles fit that alphabet. The printed faces include M21/AKH Beleren and M14 Matrix-era title typography.

The severe variant applies Gaussian blur with sigma 2.4 source pixels, then downsamples to 30% in each dimension using bilinear resampling. The transcript remains the original printed title even when blur destroys a letter or accent. This small title-only set does not establish full language, italic, rules-text, font or blur coverage.

```sh
python3 training/latin-eval/verify.py
python3 training/latin-eval/verify.py --download-sources
python3 training/latin-eval/score.py \
  --recognition-model models/ocracle-mtg-latin.rten \
  --expected-sha256 a35722103831126defa51311e17b0084fdf721b752148a74657723f551137163 \
  --output training/latin-eval/results-v1.json
```

The first command checks hashes, labels, dimensions, split membership and identity exclusion locally. The second additionally downloads each frozen source scan, checks its digest, and reproduces both derivatives. [provenance.json](provenance.json) records the acquisition context and image-library version. Card imagery/text retain Wizards of the Coast's rights; Scryfall supplies scans and localized metadata. These assets are not relicensed under the crate's MIT license.

[Fixed V1 results](results-v1.json) come from the actual Rust CLI and retain all raw predictions, scores, methods and hashes. V1 achieved **2/20 exact titles**: 1/10 clean and 1/10 severe. Character error rate was 22/330 = **6.67%**. Most errors involve accents or the German sharp s; some incorrect outputs score above 0.99. These observations do not establish accented-language support. Future independently specified checkpoints can be evaluated with a separate output file, preserving this result.
