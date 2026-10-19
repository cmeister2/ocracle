#!/usr/bin/env python3
"""Fetch only the immutable official files recorded in sources.json."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

DIRECTORY = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = json.loads((DIRECTORY / "sources.json").read_text())
    for candidate in source["candidates"]:
        for filename, record in candidate["files"].items():
            path = DIRECTORY / record["path"]
            if not path.exists() and not args.check:
                request = urllib.request.Request(record["url"], headers={"User-Agent": "ocracle-candidate-research/1"})
                data = urllib.request.urlopen(request, timeout=120).read()
                if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("Downloaded bytes disagree with pinned source: " + record["url"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if not path.exists() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Missing or corrupted candidate source: " + str(path))
            print(candidate["id"], filename, record["sha256"])


if __name__ == "__main__":
    main()
