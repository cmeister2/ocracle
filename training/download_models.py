#!/usr/bin/env python3
"""Download pinned model artifacts and verify SHA-256 before atomic placement."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        print(f'Verified {destination}')
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'ocracle-model-downloader/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'SHA-256 mismatch for {destination}: {actual}')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Downloaded and verified {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='store_true', help='Also fetch training checkpoint')
    args = parser.parse_args()
    for manifest in ['models/manifest.json', 'models/paddle-v6-manifest.json']:
        for item in json.loads((ROOT / manifest).read_text())['upstream_models']:
            if item.get('training_only') and not args.checkpoint:
                continue
            fetch(item['url'], ROOT / item['path'], item['sha256'])


if __name__ == '__main__':
    main()
