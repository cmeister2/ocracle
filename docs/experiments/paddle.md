# Official Paddle recognizer compatibility experiment

The official **PP-OCRv6 small ONNX** runs directly in RTen 0.26 and the ocracle
CLI with configurable BGR input and probability output. The original color
model's Rust integration is verified on 58 fixed crops. The earlier grayscale
ONNX adapter remains a separate reproducible comparison. No production language
profiles are registered by these experiment scripts.

`results-v6-direct.json` records all three direct-color pipelines: official
ONNX Runtime, isolated RTen, and the actual Rust CLI. Every transcript agrees
across all three. Equal-input ORT/RTen argmax sequences agree in all 58 cases;
maximum absolute probability error is 0.000025. The CLI executable, original
weights, configuration and fixture manifests are pinned by hash in that report.

| Direct color cohort | Exact | CER |
| --- | ---: | ---: |
| English gallery | 27/30 | 0.52% |
| Independent accented Latin titles | 19/20 | 0.30% |
| Small Chinese/Japanese transfer probe | 5/8 | 10.53% |
| All fixed observations | 51/58 | 0.85% |

All 29 clean observations are exact; 22/29 severe observations are exact. The
seven severe failures remain in the report. `direct-model-config.json` retains
the original BGR channels, affine normalization, minimum width and native NTC
probabilities. The real CLI uses triangle resize, while the isolated comparison
uses the publisher's OpenCV linear resize. This sample shows identical text
despite that implementation difference; it does not establish universal parity.

After installing the requirements, fetching the model, and building the runner
and root CLI as below, reproduce the direct-color integration with:

```sh
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/direct_probe.py --runner training/experimental_paddle/runner/target/release/paddle-rten-probe
```

`model-source.json` pins the publisher's [model repository and
revision](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/tree/b8f84f0b80c529de40b4fbb3544b84fa7233a513),
file digests, Apache-2.0 declaration, and primary architecture/processor sources.
The upstream license is included in `LICENSE.paddle`; downloaded weights stay
in ignored `cache/`. No weights were retrained or selected using these cases.

The source model uses BGR `[N,3,48,W]`, linear resize and normalized pixels
`(pixel/255-.5)/.5`; short lines receive zero padding to width 320. Its output is
`[N,T,18710]` **probabilities**. The dictionary has 18,708 unique single Unicode
characters, followed by ordinary space; class zero is CTC blank. The architecture
is LCNetV4 plus EncoderWithLightSVTR and a CTC inference head. The auxiliary NRTR
training head is absent from the inference graph. These details are verified
against the pinned graph/config and [official architecture
documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md).

The adapter accepts ocracle's grayscale `[N,1,48,W]` input in `[-.5,.5]`,
multiplies by two, repeats the channel three times and pads on the right when
needed. It clamps probabilities to `[1e-30,1]`, applies `Log`, then transposes to
`[T,N,C]`. It preserves all learned parameters. `model-config.json` is an
experimental multilingual template; a future registered profile must explicitly
select its card language and pin the model/configuration hashes.

On 28 frozen printed-card crops, `results-v6-small.json` records every output:

| Pipeline | Clean exact | Severe exact | Total exact | CER |
| --- | ---: | ---: | ---: | ---: |
| Official BGR, ONNX Runtime | 14/14 | 10/14 | 24/28 | 1.36% |
| Grayscale adapter, ONNX Runtime | 14/14 | 10/14 | 24/28 | 1.63% |
| Grayscale adapter, RTen raw tensors | 14/14 | 10/14 | 24/28 | 1.63% |
| Grayscale adapter, actual Rust CLI | 14/14 | 10/14 | 24/28 | 1.63% |

All 28 equal-input adapter comparisons have identical ORT/RTen argmax sequences;
the largest absolute log-probability difference is 0.000445. Input widths span
106–507 pixels, exercising both padding branches. The controlled grayscale
comparison uses integer Rec709 luminance. The CLI comparison additionally
includes its own snapshotted luminance implementation, width rounding and
triangle resize; these are separate processing paths.
The one additional grayscale character error occurs on a tiny severely degraded
Japanese title. Thus grayscale is an empirically tested compromise, not numeric
equivalence with the original color pipeline.

Twenty cases reuse the independent accented French/German/Italian/Portuguese/
Spanish title evaluation, without changing it. This model reads **19/20** exactly
(all ten clean), including `ï`, `Ä` and `ß`. The missed severe `fongoïde` becomes
`fongoide`. Two characters in the current 129-character Latin training alphabet,
`¡` and `„`, are absent from this Paddle dictionary. Complete punctuation and
MTG symbol support therefore require additional work.

Eight new cases cover Japanese title `選択`, Japanese rule `カード１枚を引く。`,
Simplified Chinese title `集群攻击` and Traditional Chinese title `劃破蒼穹`,
each clean/severe. `fixtures.json` records print IDs, source URLs/hashes, exact
bounding boxes, localized NFC truth and manual pixel review. The Japanese source
is a high-resolution scan. Both Chinese sources are marked `lowres` by Scryfall;
their visible title glyphs were reviewed, and that source limitation is retained.
Japanese title furigana is outside the main-title crop and remains untested.
Severe means Gaussian sigma 2.4 source pixels plus 30% bilinear downsampling.
Fullwidth `１` versus ASCII `1`, accents, case and punctuation remain distinct in
scoring; only NFC and whitespace are normalized.

This small probe establishes runtime compatibility and useful transfer behavior.
It does not establish all-font/all-language/full-card accuracy, MTG-specific
fine-tuning, vertical/ruby recognition, or sufficient severe-blur robustness.
The publisher's training prints are unknown, so upstream contamination cannot be
excluded. The dictionary has no Hangul, Hebrew, Arabic, Devanagari or Cyrillic
characters by Unicode-name audit. Its 64 Greek characters do not establish
Ancient Greek coverage. Confidence values remain uncalibrated.

The Japanese Opt identity appears in prior custom Latin V1 training, so its
Japanese crops are transfer/runtime evidence, not a globally unseen canonical
identity claim. Latin V2 excludes Opt. Broader Japanese and small CJK reports
overlap in some crops; their case counts must not be summed as independent data.

Reproduce from the repository root with Python 3.12+ and Rust 1.94:

```sh
python3 -m venv /tmp/ocracle-paddle-venv
/tmp/ocracle-paddle-venv/bin/pip install -r training/experimental_paddle/requirements.txt
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/fetch.py --source-scans
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/wrap.py
cargo +1.94.0 build --release --manifest-path training/experimental_paddle/runner/Cargo.toml
cargo +1.94.0 build --release
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/probe.py --runner training/experimental_paddle/runner/target/release/paddle-rten-probe
```

`fetch.py --source-scans` verifies the pinned downloads and reproduces all eight
crop/degradation PNG hashes. `probe.py` snapshots the Rust executable, records
its hash and package versions, tests ORT/RTen on identical tensors, and executes
the actual CLI once per image with only the original pass. Expected transcripts
are used solely for scoring. The arithmetic first-emission posterior in the
isolated probe differs from the CLI's geometric mean of emitted-run peaks;
neither score is a calibrated probability that the transcription is correct.

Additional frozen comparisons use the same model without fitting:

| Evaluation | Official BGR exact / CER | Actual CLI grayscale exact / CER |
| --- | --- | --- |
| English gallery: 30 crops, 15 styles/roles | 27/30 / 0.52% | 25/30 / 0.87% |
| Broader Japanese: 14 crops, title/ruby/body/flavor | 5/14 / 11.49% | 5/14 / 12.84% |

See `results-v6-gallery.json` and `results-v6-japanese.json`. Gallery color
inference reads all 15 clean crops exactly, including historic italic text and
special title treatments. Grayscale introduces an opening-quote error and a
severe Goudy word-space error. On the broader Japanese set, all five
high-resolution source crops are exact; both inherently low-resolution cases
and all seven additional severe variants fail exact scoring. Preserving color
has measurable value. These style samples remain too small to establish complete
font coverage.

Run `probe.py --scope gallery` or `probe.py --scope japanese` to reproduce each
separate report. They read existing fixtures without mutating gallery artifacts.

`additional-models.json` pins official v5 Greek/Arabic/Devanagari models and
primary-author Rivok Hebrew mobile/server CTC releases. `results-rare.json`
uses all 26 reviewed physical rare-script observations; every tested graph runs
in RTen. Exact counts are Latin(v6) 3/6, Ancient Greek(v5 Greek) 0/4, Arabic(v5)
0/4, Sanskrit(v5 Devanagari) 1/6, and Hebrew(mobile) 0/6. The stronger Hebrew
SVTRv2 CTC result in `results-hebrew-server.json` is 2/6: clean title `תהילה`
and clean keyword `יכולת תעופה` are exact; severe variants fail.

These results expose concrete structural gaps. The Greek dictionary lacks the
polytonic characters `ἘἌὄὐὸ` present in the reviewed Ancient Greek card. The
Hebrew charset lacks em dash and all niqqud/cantillation marks. The Arabic
dictionary includes all expected glyphs but contains 31 duplicate entries whose
class indices must be retained; its publisher's Arabic group reversal is
applied explicitly after CTC, with raw output preserved in the report. Whole
string reversal would corrupt embedded Latin/digit runs. No vowels/diacritics
are stripped to improve these scores. Runtime compatibility therefore does not
make these rare languages ready.

The Hebrew models are inference-only exports: their author does not publish
training weights/configurations, and states that decorative faces and vocalized
Hebrew are outside their scope. The primary-author model cards, charset and
processor revisions are pinned in the inventory. This experiment tests CTC
only; it does not implement their optional autoregressive NRTR cascade.

```sh
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/fetch.py --additional
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/rare_probe.py --runner training/experimental_paddle/runner/target/release/paddle-rten-probe
/tmp/ocracle-paddle-venv/bin/python training/experimental_paddle/rare_probe.py --runner training/experimental_paddle/runner/target/release/paddle-rten-probe --hebrew-model server --only-language he --output training/experimental_paddle/results-hebrew-server.json
```
