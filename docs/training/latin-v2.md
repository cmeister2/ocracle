# Latin V2 training experiment

`ocracle-mtg-latin-v2` is an experimental CRNN trained for English, French, German, Italian, Portuguese and Spanish card text. It improves the frozen V1 accent experiment but remains **unregistered** because physical recognition is still inconsistent. These results do not establish support for every printed language, font or damage condition.

The fixed independent physical accent benchmark uses ten card identities absent from the known training lineage, with clean and severe derivatives. Actual Rust inference, with all preprocessing passes, produced:

| Condition | V1 exact | V2 exact |
| --- | ---: | ---: |
| Clean | 1/10 | 7/10 |
| Severe blur + downsampling | 1/10 | 5/10 |
| All | 2/20 | 12/20 |

Overall CER fell from 6.67% to 2.73%. German and Portuguese were exact in all four observations each; French was exact in none of four, and Italian in one of four. For example, V2 still reads `Renaissance fongoïde` as `Renaissance fongoide` and can hallucinate an accent in `Opportunità`. Raw candidates, confidence scores, configuration and executable hashes are preserved in [the independent results](../../training/latin-eval/results-v2.json). Confidence is an uncalibrated model score, not a probability of an exact transcription.

A separate reviewed physical test contains 20 title crops from 18 canonical identities. Each is scored in clean, Gaussian-severe and combined degradation conditions, after validation-only checkpoint selection:

| Recognizer | CER | Exact observations |
| --- | ---: | ---: |
| Generic English source | 7.91% | 0/60 |
| Latin V1 | 6.71% | 6/60 |
| Latin V2 | 4.48% | 22/60 |

This small test includes only one Italian title. Internal severe Gaussian blur and the independent benchmark's blur-plus-downsampling condition differ. The complete [training report](../../training/latin-v2-report.json) records the fixed selection score, all five validation checkpoints, held-out outputs and loss history. No independent test label selected the checkpoint.

The frozen English physical-font gallery was also evaluated after selection, without modifying the public gallery: V2 scored 18/30 exact and 5.90% CER. The frozen Ocrs English baseline scored 19/30 exact and 6.08% CER. The two metrics move differently, so there is no blanket font-improvement claim. The default [Paddle colour baseline](../experiments/paddle.md) is evaluated separately and is not replaced by this CRNN. [Raw V2 observations](../../training/latin-v2-english-gallery.json) preserve every prediction and preprocessing candidate.

The corpus has 43,145 lines, including 101 newly reviewed localized title crops and 20 usable reviewed body lines. All 63 English fallback images discovered in localized records were rejected before training. Procedural CLDR exemplar practice provides positive training observations for the standard Latin orthography and typographic punctuation targets; physical examples remain sparse. The [corpus documentation](../../training/latin-v2-corpus/README.md), [coverage audit](../../training/latin-v2-corpus/coverage.json) and versioned contact sheets distinguish actual scans, synthetic card text and glyph practice.

Training used 1,000 CPU updates, a 150-update head warm-up, then upper-convolution/GRU fine-tuning with a lower learning rate and clean-English teacher preservation. All 12 batch-normalization buffers remained unchanged. The selected model's convolutional and recurrent parameter changes are recorded in the report. The exact [V1 PyTorch initializer](../../models/training/latin-v1-source.json) is pinned for reproduction. Opt is excluded from V2 updates but appeared in that earlier initializer; it is not an unseen identity throughout the complete lineage.

The 214-character NFC alphabet, 64-pixel input height and per-card language are declared in `latin-v2-profiles/{en,fr,de,it,pt,es}.json`. The shared model consumes normalized grayscale lines and emits 215-class CTC log probabilities. Explicit experimental invocation:

```sh
target/release/ocracle recognize IMAGE --line --passes all \
  --recognition-model models/ocracle-mtg-latin-v2.rten \
  --model-config models/latin-v2-profiles/fr.json
```

[The artifact manifest](../../models/latin-v2-manifest.json) pins weights, configuration hashes and provenance. The weights retain Robert Knight/`ocrs-models` attribution and [CC-BY-SA-4.0](../../models/LICENSE-CC-BY-SA-4.0.txt). Card content remains Wizards of the Coast material; font and source-image rights are documented with the corpus. Reproduction verifies the entire labels-and-pixels manifest before training. Export checks cover dynamic widths/batches and actual Rust transcripts in all six profiles.

## Data provenance and reproduction

This corpus belongs to the separate `ocracle-mtg-latin-v2` CRNN experiment. It does not replace the frozen English or Latin V1 reports, and its profiles are not registered for automatic use.

`cards.json` contains localized Scryfall metadata from the dated all-cards snapshot identified in `selection.json`. Candidate selection uses canonical card-name hash splits, accent coverage and frame cohorts. Recognition output does not filter the new title candidates. `orthography.json` pins CLDR release 48 main and auxiliary exemplar sources and their hashes; it describes a training alphabet target, not demonstrated accuracy.

`titles-pending.json` records candidate pixels, source URLs and hashes. `titles-reviewed.json` records every accepted or rejected title after visible text, complete crop bounds, case and diacritics were checked: 101 accepted titles and 63 rejected English fallbacks. The reviewed [contact sheets](../../training/latin-v2-corpus/contact-sheets/manifest.json) are versioned, including rejection evidence. Older localized records can point to **English fallback scans**, including through Gatherer multiverse IDs. Those records are rejected as physical labels. A localized metadata field or source ID alone is insufficient evidence of the image language. `unavailable-physical.json` also records 11 placeholders with no alternative source.

`body-lines.json` contains the 21 accepted manually reviewed localized body crops, including italic reminder/flavor lines. Twenty remain after excluding the subsequently reserved Opt identity. A German proposal with an omitted final period is excluded. Spanish reminder annotations include the reviewer's italic corrections. Body candidates were discovered by accent-folded metadata matching, so these lines have a separate selection bias from the unfiltered titles.

The materialized corpus combines the frozen English and Latin V1 samples, accepted new physical lines, new synthetic card text in open fonts, and explicitly marked procedural glyph practice. All genuine card identities retain their original train/validation/test split across languages and printings. The ten independent physical accent-test identities, gallery cards, rare-script singletons and Japanese Opt holdout are excluded. Procedural glyph practice is training-only and is never counted as physical evaluation.

The finalized corpus contains 43,145 lines and a 214-character head. It excludes 139 Opt rows inherited from the parent datasets. The pinned V1 initialization checkpoint already trained on Opt, so this exclusion applies to V2 updates; it does **not** establish that Opt is unseen throughout the V1→V2 model lineage. The independent Latin accent test and five rare-script singleton identities are disjoint throughout the known lineage.

`summary.json` freezes counts and the manifest SHA-256. Every materialized manifest row includes its PNG hash, so replay verifies both labels and pixels, including references to the parent corpora. Reproduction regenerates the synthetic data with the pinned fonts and seed and downloads hash-pinned accepted scans for deterministic recropping. Model training has fixed initialization, schedule and validation selection; bitwise-identical weights across different hardware or PyTorch builds are not promised.

The exact Latin V1 PyTorch initializer is versioned at [models/training/latin-v1-source.pt](../../models/training/latin-v1-source.pt), with a [hash and provenance manifest](../../models/training/latin-v1-source.json). The original generic checkpoint used by the English teacher is fetched and digest-checked by `download_models.py`. The full Scryfall bulk snapshot is needed only to choose a new acquisition cohort; replay uses the frozen `cards.json` and does not need to download the roughly 400 MB bulk file.

Run with the pinned environment described in [the training instructions](../../training/README.md):

```sh
python training/latin_v2_reproduce.py --skip-train
python training/latin_v2_train.py --steps 1000 --threads 4
python training/latin_v2_validate.py
```

Running `python training/latin_v2_reproduce.py` without `--skip-train` performs the full data replay, training and validation workflow. `latin_v2_validate.py` scores both source models on the fixed physical test, converts the selected ONNX checkpoint to RTen, verifies dynamic tensors and actual Rust transcripts for six profiles, evaluates the independent accents and English font gallery, then writes the digest-pinned experimental artifact manifest. These post-selection evaluations never feed back into checkpoint choice.

Training preserves batch-normalization running statistics, warms up the Unicode head for 150 updates, then updates the upper convolutional layers and both GRUs with a lower learning rate. Every update includes a clean English training line and a KL penalty against the original generic recognizer. Checkpoints are selected using only fixed validation card identities: physical CER and exactness plus English preservation. Test labels are evaluated after selection. The independent physical accent benchmark is scored separately through the Rust runtime.

The [coverage audit](../../training/latin-v2-corpus/coverage.json) separates actual physical observations from procedural glyph practice. The training set contains only three physical Ä occurrences, four ï, two Ê, two È and one ù. Some other exemplar characters occur only in synthetic practice. The inherited general alphabet also retains punctuation classes with no new MTG labels. These counts establish data coverage, not reliable physical recognition. Internal `severe` evaluation applies Gaussian blur with sigma 1.6–2.8 at the native crop resolution; `combined` applies downsampling, blur and JPEG. The independent accent benchmark's severe derivative combines blur and downsampling and is a different condition.

Card text/images remain Wizards of the Coast material obtained through Scryfall and Gatherer, used here for noncommercial fan OCR research. They are not relicensed as the Rust crate. URW fonts carry AGPL-3.0 with the font exception; DejaVu font terms are frozen in `DEJAVU-LICENSE.txt`. Model derivatives retain the upstream recognizer's CC-BY-SA-4.0 attribution, recorded in the model documentation.
