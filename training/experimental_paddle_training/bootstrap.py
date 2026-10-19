#!/usr/bin/env python3
"""Fetch only bounded, digest-pinned official native training dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def fetch(record, destination):
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(record["url"], headers={"User-Agent": "ocracle training research"})
        temporary = destination.with_suffix(destination.suffix+".part")
        try:
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as output:
                count = 0
                digest = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > record["bytes"]:
                        raise RuntimeError("Download exceeds pinned size: " + str(destination))
                    output.write(chunk)
                    digest.update(chunk)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise RuntimeError("Pinned download differs: " + str(destination))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    data = destination.read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise RuntimeError("Pinned source/checkpoint differs: " + str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    source = json.loads((HERE / "source-files.json").read_text())
    for record in source["files"]:
        fetch(record, HERE / "vendor" / record["path"])
    if not args.source_only:
        checkpoint = json.loads((HERE / "source-checkpoint.json").read_text())
        fetch(checkpoint, HERE / checkpoint["path"])
    print("Verified pinned official source" + ("" if args.source_only else " and native checkpoint"))


if __name__ == "__main__":
    main()
