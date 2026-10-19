# Official Paddle candidates

Frozen development comparison of official PP-OCRv6 medium and PP-OCRv5 Latin against the small baseline. The [fresh comparison](../../docs/experiments/paddle-blur.md) supports the current six-language medium original-pass default; historical observations remain unchanged.

See [method, source pins, limitations and results](../../docs/experiments/paddle-candidates.md).

```sh
python3 training/experimental_paddle_candidates/fetch.py
python3 training/experimental_paddle_candidates/compatibility.py --runner /path/to/paddle-rten-probe
python3 training/experimental_paddle_candidates/evaluate.py
python3 -S training/experimental_paddle_candidates/evaluate.py --check
```
