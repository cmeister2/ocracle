# Official Paddle candidates

Isolated development comparison of official PP-OCRv6 medium and PP-OCRv5 Latin against the current small model. Production profiles and frozen tests are unchanged.

See [method, source pins, limitations and results](../../docs/experiments/paddle-candidates.md).

```sh
python3 training/experimental_paddle_candidates/fetch.py
python3 training/experimental_paddle_candidates/compatibility.py --runner /path/to/paddle-rten-probe
python3 training/experimental_paddle_candidates/evaluate.py
python3 -S training/experimental_paddle_candidates/evaluate.py --check
```
