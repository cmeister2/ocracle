# Poster-lettering challenge

Five complete printed Secret Lair title fields, each clean/severe, frozen before
OCR and excluded from custom training. [Method, results and limitations](../../docs/evaluation/poster-lettering.md).

Use Python3.12+ with Pillow13.0.0 and the built Rust CLI:

```sh
python scripts/verify_models.py --require-models
python training/style-eval/prepare.py --check --download-sources
python training/style-eval/score.py --output /tmp/ocracle-style-results.json
python training/style-eval/score.py --detected --output /tmp/ocracle-style-detected-results.json
```
