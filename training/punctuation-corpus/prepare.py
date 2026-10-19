#!/usr/bin/env python3
"""Reproduce three independently reviewed, training-only punctuation crops."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import unicodedata
import urllib.request

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(value):
    """Collect protected canonical names and identifiers from fixture metadata."""
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"name", "card_name", "oracle_id", "card_id", "print_id"} and isinstance(child, str):
                result.add(child.casefold())
            result.update(identities(child))
    elif isinstance(value, list):
        for child in value:
            result.update(identities(child))
    return result


def protected_identities():
    protected = set()
    paths = ["docs/gallery/assets/fixtures.json", "training/latin-eval/manifest.jsonl",
             "training/style-eval/cards.json", "training/rare-scripts/cards.json",
             "training/experimental_easyocr/fixtures.json", "training/blur-corpus/cards.json"]
    for name in paths:
        path = ROOT / name
        # The fresh blur corpus is developed in a separate stacked PR.
        if name == "training/blur-corpus/cards.json" and not path.exists():
            continue
        values = ([json.loads(line) for line in path.read_text().splitlines()]
                  if path.suffix == ".jsonl" else json.loads(path.read_text()))
        protected.update(identities(values))
    return protected


def check(replay=False, download=False):
    annotations_path = DIRECTORY / "annotations.json"
    cards_path = DIRECTORY / "cards.json"
    provenance_path = DIRECTORY / "provenance.json"
    rows = json.loads(annotations_path.read_text())
    cards = {card["id"]: card for card in json.loads(cards_path.read_text())}
    if len(rows) != 3 or len(cards) != 3:
        raise ValueError("Expected exactly three reviewed training sources")
    pins = json.loads(provenance_path.read_text())
    if pins["annotations_sha256"] != sha(annotations_path) or pins["cards_sha256"] != sha(cards_path):
        raise ValueError("Frozen punctuation annotations or source metadata changed")
    protected = protected_identities()
    counts = Counter()
    sources_replayed = 0
    for row in rows:
        card = cards[row["card_id"]]
        if any(row[key] != card[field] for key, field in
               [("card_name", "name"), ("oracle_id", "oracle_id"), ("language", "lang")]):
            raise ValueError("Training crop identity differs from source metadata")
        bucket = int(hashlib.sha256(row["card_name"].casefold().encode()).hexdigest()[:8], 16) % 10
        if bucket < 2 or row["split"] != "train":
            raise ValueError("Validation/test identity cannot become punctuation training data")
        if {row["card_name"].casefold(), row["card_id"], row["oracle_id"]} & protected:
            raise ValueError("A protected evaluation identity overlaps punctuation training")
        if (row["review_status"] != "accepted" or not row["italic"]
                or row["text"] != unicodedata.normalize("NFC", row["text"])
                or any(value is not True for key, value in row["independent_review"].items() if key != "reviewer")):
            raise ValueError("Missing literal transcription or independent review")
        left, top, right, bottom = row["bbox"]
        width, height = row["source_size"]
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("Crop bounds are outside the source image")
        path = DIRECTORY / row["path"]
        image = path.read_bytes()
        if (sha(path) != row["image_sha256"] or image[:8] != b"\x89PNG\r\n\x1a\n"
                or struct.unpack(">II", image[16:24]) != (right - left, bottom - top)):
            raise ValueError("Frozen punctuation crop bytes or dimensions changed")
        counts.update(character for character in row["text"] if character in "¡„ẞ")
        source = ROOT / "training/data/latin/scans" / (row["card_id"] + ".jpg")
        if not source.exists() and download:
            source.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(row["source_url"], headers={"User-Agent": "ocracle-punctuation-replay/1"})
            source.write_bytes(urllib.request.urlopen(request, timeout=60).read())
        if source.exists():
            if sha(source) != row["source_sha256"]:
                raise ValueError("Original source hash mismatch")
            if replay:
                from PIL import Image, __version__
                if __version__ != "13.0.0":
                    raise ValueError("Pillow13.0.0 is required for exact source replay")
                import io
                original = Image.open(source).convert("RGB")
                buffer = io.BytesIO()
                original.crop(row["bbox"]).save(buffer, format="PNG")
                if list(original.size) != row["source_size"] or buffer.getvalue() != image:
                    raise ValueError("Source pixels do not reproduce the reviewed crop")
                sources_replayed += 1
        elif replay:
            raise ValueError("Fresh replay needs original source scans; pass --download-sources")
    if counts != Counter({"„": 2, "¡": 1}):
        raise ValueError("Expected two physical low quotes and one inverted exclamation")
    result = {"training_lines": 3, "physical_target_occurrences": dict(counts),
              "sources_replayed": sources_replayed, "protected_identity_overlap": False}
    print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify frozen evidence without changing it")
    parser.add_argument("--replay", action="store_true", help="Additionally reproduce exact PNG bytes from original sources")
    parser.add_argument("--download-sources", action="store_true")
    args = parser.parse_args()
    check(replay=args.replay, download=args.download_sources)


if __name__ == "__main__":
    main()
