# Official Paddle recognizer candidates

On a fixed Latin development cohort, unmodified PP-OCRv6 medium improves original-pass recognition over the small baseline, particularly on italic and blurred lines. This document preserves that candidate experiment and its original observations; the current default decision is supported by the separate fresh comparison.

The [fresh blur comparison](paddle-blur.md) supports medium with one original-image pass as the default for `en`, `es`, `fr`, `de`, `it` and `pt`. Normal callers select only the printed language; `la` and best-effort `ja`, `zhs` and `zht` retain small. Clean-data regressions qualify the aggregate improvement. In the fixed gallery, medium reads `Renaissance fongoïde` as `Renaissance fongoide` on both clean and blurred inputs; small read the clean case correctly. These failures remain in the recorded results.

## Sources and selection

Two official candidates were selected before inference: the larger current v6 recognizer and the current Latin-specific v5 recognizer. Paddle lists medium as its largest v6 recognizer, with LCNetV4 and LightSVTR; its internal printed-English and artistic results exceed small. Those publisher metrics motivate the experiment and do not establish MTG performance. [Pinned official model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/blob/50c7eacafc52fa7bcf4194e8cd08e46f8558504b/README.md).

Paddle also publishes a Latin-specific v5 model for Spanish, French, Portuguese, German, Italian and other Latin-script languages. Its separate benchmark is not directly comparable with v6's benchmark. The current official model listing contains v6 tiny/small/medium and Latin-specific v3/v5; no separate official Latin-only v6 checkpoint was found on 2026-10-19. [Official multilingual v5 documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md), [publisher's Latin model listing](https://huggingface.co/models?search=latin&author=PaddlePaddle).

| Recognizer | Original ONNX bytes | Immutable Hugging Face revision | ONNX SHA-256 |
|---|---:|---|---|
| v6 small, baseline | 21,159,378 | `b8f84f0b80c529de40b4fbb3544b84fa7233a513` | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |
| v6 medium | 76,554,979 | `50c7eacafc52fa7bcf4194e8cd08e46f8558504b` | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |
| v5 Latin mobile | 8,042,023 | `89d3a50e2c27e2e7cceeab0e944c25c807d5db4f` | `7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498` |

Both candidate graphs are below the 200 MB research limit. Original publisher files and dictionaries are downloaded from revision-pinned URLs, without conversion or weight changes. Apache-2.0 is declared by the official model repositories; the pinned PaddleOCR license is included with the source evidence. [Medium repository](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/tree/50c7eacafc52fa7bcf4194e8cd08e46f8558504b), [Latin repository](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx/tree/89d3a50e2c27e2e7cceeab0e944c25c807d5db4f), [source pins](../../training/experimental_paddle_candidates/sources.json).

## Character and tensor contracts

The audited dictionaries preserve every original class index. Class zero is CTC blank, followed by the YAML dictionary and an appended ordinary space. All dictionary entries are single Unicode scalars.

| Dictionary | Entries | Unique entries | Classes including blank and appended space | ¡ | „ | ẞ | ß / ï / Ä |
|---|---:|---:|---:|---|---|---|---|
| v6 small and medium | 18,708 | 18,708 | 18,710 | absent | absent | absent | present |
| v5 Latin | 836 | 772 | 838 | absent | present | absent | present |

Medium's alphabet is identical to small, UTF-8 SHA-256 `d9a64a84befbb1f2e77ab4f2454acdb9cf1995e96567555aa6c30c2256d0fecb`. The v5 alphabet hash is `b0935f2331b3507d91ee1cf4af0c486d69788425fdaf7c242cce153fdc2f7275`. Its 64 duplicated characters retain distinct output classes; deduplicating them would corrupt decoding. Neither candidate can emit ¡ or ẞ. This cohort contains no unsupported reference characters, so its scores do not measure these structural gaps. [Dictionary audit with exact class indices](../../training/experimental_paddle_candidates/dictionary-audit.json).

Both graphs accept dynamic-width `float32[N,3,48,W]` BGR pixels, scaled by `2/255` with offset `-1`, aspect-preserving ceil width and normalized-zero right padding to at least 320. Outputs are `float32[N,T,C]` softmax probabilities; medium uses ONNX opset 11 and Latin v5 uses opset 7. Production CLI resizing uses Rust image's Triangle filter, while the isolated official-input parity probe uses OpenCV linear resizing. [Pinned configs](../../training/experimental_paddle_candidates/configs/), [official preprocessing contract already audited for the original Paddle experiment](../../training/experimental_paddle/model-source.json).

All nine raw-tensor probes execute unmodified in RTen 0.26: each of the three models receives the first development title's clean variant, the first italic line's downsample variant, and a normalized-zero width-320 tensor. Their selection precedes OCR and is encoded in the script. Every ORT/RTen argmax sequence agrees; maximum absolute probability difference is 0.000312. The zero input checks runtime/padding behavior and is excluded from accuracy denominators. [Compatibility report](../../training/experimental_paddle_candidates/compatibility.json).

## Fixed development comparison

The input is every accepted LatinV2 validation title (23) and every reviewed non-Opt body line (20), with four predefined RGB variants: clean, Gaussian sigma 2.4, Gaussian followed by 30% downsampling, and then JPEG quality 45. This gives 172 inputs per model and 516 actual Rust CLI calls. Labels, source hashes, crop bounds and all variants were fixed before this candidate inference. All calls succeed; every raw JSON result, confidence and stderr is retained. The snapshot executable SHA-256 is `1ede5e7d7a8c2060114ab471adf040d965d0e697eb55f8d8eb094a66b6b28e13`. [Development fixtures and source replay](../../training/experimental_colour/fixtures.json), [complete candidate report](../../training/experimental_paddle_candidates/results-colour-development.json).

Only `--line --passes original` is used. There is no confidence selection, fine-tuning, image adaptation or reference-dependent crop selection. Scoring applies NFC and collapses whitespace; case, punctuation and accents remain significant.

| Input group | Observations per model | Small exact | Medium exact | Latin v5 exact |
|---|---:|---:|---:|---:|
| All | 172 | 92 | 104 | 98 |
| Clean | 43 | 39 | 39 | 40 |
| Gaussian | 43 | 24 | 28 | 28 |
| Downsampled | 43 | 20 | 27 | 21 |
| JPEG | 43 | 9 | 10 | 9 |
| Italic, across variants | 36 | 17 | 26 | 18 |
| Regular, across variants | 136 | 75 | 78 | 80 |

Overall character error rate is **8.062% small, 6.732% medium, 8.233% Latin v5**. Medium corrects 15 small-model failures and regresses three: one Gaussian and two JPEG cases. Clean exactness is unchanged on the same cases. Heavy JPEG damage remains difficult. The Latin-specific model improves clean exactness but has a worse overall character error rate, so its language focus alone does not justify replacing the current recognizer.

Every call records elapsed process wall time, including model loading and startup. The medians are small 0.180 s, medium 0.387 s and Latin v5 0.123 s. Two concurrent calls each use two Rayon threads, with other work on the host. These measurements describe this experiment, not controlled throughput or steady-state inference. Medium's weight file is 3.62 times larger than small.

These are development/training identities with known CRNN lineage. Original body discovery used accent-folded OCR matching before manual label review, which creates selection bias. The languages are French, German, Italian, Portuguese and Spanish; this is not an English historic-font inventory. Upstream Paddle training membership is unknown. No frozen English gallery, independent accented-title, rare-script or concert-poster test was used for candidate selection. The [separate fresh evaluation](paddle-blur.md) supplies card-disjoint evidence and retains both original/enhanced outputs for the default decision; these line-crop runs do not establish whole-card layout accuracy. No all-font or all-language completion claim follows from this result.

## Reproduction

Use Python 3.12 and the [pinned experiment requirements](../../training/experimental_paddle_candidates/requirements.txt). Prepare the existing RGB development fixtures using their [replay script](../../training/experimental_colour/evaluate.py), and build the Rust CLI and the [isolated RTen runner](../../training/experimental_paddle/runner/). Downloads require network access; checks of frozen reports do not.

```sh
python3 -m venv /tmp/ocracle-paddle-candidates
/tmp/ocracle-paddle-candidates/bin/pip install -r training/experimental_paddle_candidates/requirements.txt
python3 training/download_models.py
python3 training/experimental_paddle_candidates/fetch.py
cargo build --release
cargo build --release --manifest-path training/experimental_paddle/runner/Cargo.toml --target-dir /tmp/ocracle-paddle-target
/tmp/ocracle-paddle-candidates/bin/python training/experimental_paddle_candidates/compatibility.py --runner /tmp/ocracle-paddle-target/release/paddle-rten-probe
python3 training/experimental_paddle_candidates/evaluate.py
python3 -S training/experimental_paddle_candidates/evaluate.py --check
```

`evaluate.py` snapshots the executable and all model weights before its first call, verifies image hashes, and rejects a manifest change during the run. Its offline checker binds all 516 observations to the source model pins, configurations, languages, fixture hashes, raw inference JSON and recomputed metrics; it needs no model cache or executable. Re-running evaluation writes only this experiment's development report and ignored cache.
