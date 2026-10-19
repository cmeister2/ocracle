# Paddle small versus medium on fresh printed blur lines

**Paddle small remains the default.** Medium is available through six opt-in profiles: `en-medium`, `es-medium`, `fr-medium`, `de-medium`, `it-medium`, and `pt-medium`. Medium reduces character errors on this development cohort, especially strong blur, but loses exact clean lines and some literal punctuation. Latin `la` and best-effort CJK profiles remain on small; this comparison provides no evidence for changing them.

The [fresh corpus](../evaluation/blur-development.md) was frozen before inference: 16 canonical identities absent the existing project corpora/evaluations, 64 independently pixel-reviewed native lines, seven fixed conditions, 448 observations per model. There are 36 italic base lines. Upstream pretrained membership is unknown. This is development evidence and may inform model choice, not an untouched test result.

The two official, unmodified checkpoints were fixed in advance: small SHA256 `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` (21,159,378 bytes) and medium `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` (76,554,979 bytes). Both use their pinned official dictionary/tensor configurations and the card's actual language. No weights were fitted and no observations were discarded.

## Results and tradeoffs

CER is total Levenshtein edits divided by total reference characters after NFC and whitespace normalization only. Case, accents and printed punctuation remain literal. `Selected` is the production maximum-confidence choice between original and enhanced passes, with the last candidate winning ties. Confidence is the uncalibrated geometric mean of emitted CTC character posterior peaks, not an accuracy probability.

| Model | Pass | Exact / 448 | CER |
| --- | --- | ---: | ---: |
| Small | Original | 313 | 4.51% |
| Small | Enhanced | 305 | 6.17% |
| Small | Selected | 313 | 4.55% |
| Medium | Original | 314 | 2.78% |
| Medium | Enhanced | 296 | 3.57% |
| Medium | Selected | 308 | 2.79% |

Medium lowers selected CER from 4.55% to 2.79%, while selected exact lines fall from 313 to 308. On the 384 degraded observations, CER falls from 5.27% to 3.16%. On the 64 clean observations, CER rises from 0.23% to 0.55% and exact lines fall from **59 to 52**. Mild Gaussian blur and 50% downsampling also favor small on exact lines. The advantage on strong degradation does not imply every crop improves.

| Condition | Small exact / 64 | Small CER | Medium exact / 64 | Medium CER |
| --- | ---: | ---: | ---: | ---: |
| Clean | 59 | 0.23% | 52 | 0.55% |
| Gaussian σ 1.2 | 55 | 0.42% | 49 | 0.78% |
| Gaussian σ 2.4 | 41 | 1.52% | 43 | 1.29% |
| Downsample 50% | 55 | 0.51% | 48 | 0.83% |
| Horizontal motion 9 px | 14 | 23.96% | 18 | 13.90% |
| JPEG quality 25 | 60 | 0.23% | 61 | 0.14% |
| Severe: Gaussian σ 2.4 + downsample 30% | 29 | 4.99% | 37 | 2.03% |

| Language | Observations | Small selected exact / CER | Medium selected exact / CER |
| --- | ---: | ---: | ---: |
| German | 84 | 60 / 3.98% | 68 / 1.69% |
| English | 112 | 94 / 2.46% | 95 / 1.17% |
| Spanish | 56 | 28 / 7.79% | 31 / 4.09% |
| French | 84 | 51 / 3.51% | 37 / 2.91% |
| Italian | 56 | 36 / 5.80% | 34 / 4.63% |
| Portuguese | 56 | 44 / 5.41% | 43 / 3.47% |

On the 252 italic observations, selected CER falls from 5.62% to 3.63%, while exact lines fall from 154 to 143. On 196 regular/title observations, CER falls from 2.50% to 1.17% and exact lines rise from 159 to 165. Full breakdowns by style, condition and language are retained in the report.

Several clean regressions are literal punctuation substitutions. For example, the Italian reference `Un’oscurità che inghiotte la luce.` becomes `Un'oscurità che inghiotte la luce.` in medium, at confidence 0.9998717. A Portuguese clean line loses the accent in `talídias`. These differences remain errors; punctuation/accent folding was not added after seeing results. The visibly confirmed German low opening quote `„` is absent from both v6 dictionaries and remains in the ground truth.

Confidence selection also makes mistakes. Compared with original alone, selected small helps five exact lines and harms five; selected medium helps two and harms eight. Enhanced is selected on 113 small observations and 167 medium observations. Every candidate and its raw confidence remain inspectable.

## Decision protocol

The numerical eligibility gates were recorded **after the evaluation was launched but before any predictions, metrics or progress log were inspected**: selected medium CER must decrease on all 448 cases and on the 384 degraded cases; clean CER may rise at most 0.5 percentage points; no measured language may rise more than 1 percentage point. Exact match counts are secondary to these gates, and the larger download is acceptable if they pass. No frozen gallery/accent/rare/poster tests were used to choose or tune candidates.

All four eligibility gates pass: overall CER changes by −1.761 percentage points, degraded CER by −2.108 points, clean CER by +0.323 points, and every measured language improves CER. Eligibility does not require replacing the default. The final decision retains small by default because the clean exact-line loss and punctuation regressions matter, and exposes medium as an explicit alternative for the six measured languages. No new preprocessing, threshold or model weights were tuned against this corpus.

## Reproduce and inspect

[evaluate_blur.py](../../training/experimental_paddle_candidates/evaluate_blur.py) and [results-blur-development.json](../../training/experimental_paddle_candidates/results-blur-development.json) retain all **896 actual CLI processes**, both candidates, selected output, raw stdout/stderr and their digests, literal references, exact image/source/configuration/model/CLI hashes, recorded commands and process timings. The immutable production CLI hash is `1ede5e7d7a8c2060114ab471adf040d965d0e697eb55f8d8eb094a66b6b28e13`. It uses native BGR originals and the existing grayscale enhancement; discarded experimental RGB enhancement is absent.

```sh
python3 training/experimental_paddle_candidates/evaluate_blur.py --check
```

Offline checking invokes no model and requires no downloaded weights. It validates the exact 2 × 448 Cartesian membership, frozen references/images, source/configuration pins, raw inference binding, finite confidences, both pass methods, last-on-tie maximum selection, all metric aggregates and fixed-rule assessment. If local model/CLI snapshots are present, their bytes are checked too.

To repeat inference after fetching the pinned candidate weights, supply the exact predeclared executable snapshot:

```sh
python3 training/experimental_paddle_candidates/evaluate_blur.py \
  --cli /path/to/ocracle-snapshot
```

The evaluator runs two concurrent CLI processes with `RAYON_NUM_THREADS=2`, using `--line --passes all`. There was one evaluation run; adding the declared decision metadata afterward preserved the raw 896-record bundle SHA256 `a78a252906c2fafa5eff56b09561d5346e4c05e111b4a5f12cb6278c2b8b0336` without rerunning inference.

The run took 174.14 seconds. Median per-process wall time was 0.228 seconds for small and 0.549 seconds for medium; p95 was 0.305 and 0.663 seconds. These include separate-process model loading and both passes under concurrent CPU load. They are process timings, not a throughput benchmark or a promise about other hardware.
