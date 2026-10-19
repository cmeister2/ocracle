# Colour enhancement development experiment

Preserving RGB through contrast stretching and sharpening did **not** improve the
default recognizer on this development cohort. Both implementations read 95/172
observations exactly; selected CER changes from 8.08% to 8.10%. The tested patch
remains experimental. The runtime and public gallery remain unchanged.

| Actual Rust output | Existing enhancement exact / CER | RGB enhancement exact / CER |
| --- | ---: | ---: |
| Original pass | 92/172 / 8.06% | 92/172 / 8.06% |
| Enhanced pass | 101/172 / 9.13% | 96/172 / 11.21% |
| Confidence-selected | 95/172 / 8.08% | 95/172 / 8.10% |

The original candidates are byte-for-byte identical in all 172 pairs. The RGB
variant worsens enhanced-pass CER and exactness, even though per-line confidence
selection hides much of the difference. A higher exact count and a lower total
CER need not move together. No reference text selects a candidate.

The 43 lines are all 23 accepted Latin V2 validation titles and all 20 reviewed
body lines after excluding Opt. They cover French, German, Italian, Portuguese
and Spanish; nine lines are italic. Selection was fixed before this comparison
and includes every eligible line. Body discovery originally used accent-folded
metadata matching, so it has that selection bias. These are known development
and training identities from the weaker CRNN lineage, not an independent test
or comprehensive font sample. Paddle upstream pretraining membership is unknown.
Frozen gallery, accent, rare-script and poster tests did not select this change.

[The manifest](../../training/experimental_colour/fixtures.json) pins manual
review sources, source scan hashes, localized text and complete crop bounds.
The RGB pixels are recropped from those exact original sources after 672×936
bicubic resizing. The original training crops were grayscale; no colour is
invented from them. Each RGB crop has four fixed variants:

1. Clean crop.
2. Gaussian blur with sigma 2.4 source pixels.
3. That blur followed by 30% bilinear downsampling in each dimension.
4. The downsampled crop JPEG-encoded at quality 45 with 4:2:0 chroma subsampling.

All 172 PNGs are committed and hash-checked. Pillow 13.0.0 is required to reproduce
their exact bytes. These controlled corruptions expose robustness limits;
they are not a distribution of real camera blur. The JPEG condition is hard:
the selected existing model is exact on 9/43, versus 38/43 clean.

[results.json](../../training/experimental_colour/results.json) retains both
complete inference outputs, both preprocessing candidates, per-line uncalibrated
confidence, executable/configuration/model digests and group metrics. Both runs
use identical official PP-OCRv6 small weights and registered language profiles.
Configurations, weights and binaries are snapshotted before inference. Input
PNGs are hash-checked before inference and again during offline verification.
Scoring preserves case and punctuation, applying NFC and collapsed whitespace.

The [archived patch](../../training/experimental_colour/colour-preserving.patch)
keeps colour through a shared luminance-derived contrast stretch and RGB unsharp
mask for colour model inputs. Its grayscale branch retains the original
operation order. The tested variant passed stable/MSRV checks and a regression
that uniform coloured input retains distinct RGB channels; that test is in the
patch rather than the production runtime. [source.json](../../training/experimental_colour/source.json)
pins the baseline revision, before/after engine hashes and exact patch. A second
full replay built both variants with Rust 1.99 and reproduced every raw inference
record and metric exactly; [reproduction.json](../../training/experimental_colour/reproduction.json)
records the two binary hashes and binds the original report.

To check frozen evidence without inference, models or third-party Python modules:

```sh
python3 -S training/experimental_colour/evaluate.py --check
```

To reproduce the experiment, create a separate checkout of the exact baseline
revision in `source.json`, build its CLI and copy that executable aside. Copy
this experiment directory into that checkout. Fetch the pinned model files with
`training/download_models.py`, then apply `colour-preserving.patch` from the
checkout root and rebuild. Using the pinned Pillow environment:

```sh
python training/experimental_colour/evaluate.py --prepare
python training/experimental_colour/evaluate.py --before /ABS/PATH/BASELINE-CLI
```

Fresh inference writes `/tmp/ocracle-colour-results.json` by default; `--output`
selects another path without replacing the frozen report. Preparation reuses or
downloads the original source scans and rejects a digest
mismatch. Repeated preparation must match the frozen manifest exactly. A fresh
run requires the patched engine hash, records its own executable hashes and
recomputes every metric. The offline checker verifies all case membership,
image/configuration/model bindings, selected-candidate provenance and metrics.
Original card imagery/text remains Wizards of the Coast material, with source
links retained; the corpus is not relicensed as MIT crate source.
