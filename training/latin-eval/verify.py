#!/usr/bin/env python3
"""Verify independent title evidence and, optionally, its original scans."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import unicodedata
import urllib.request

from PIL import Image, ImageFilter

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def digest(data):
    """Identify the exact bytes of a frozen source or crop."""
    return hashlib.sha256(data).hexdigest()


def main():
    """Check local evidence without consulting any recognizer or prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-sources", action="store_true")
    args = parser.parse_args()
    cards = json.loads((DIRECTORY / "cards.json").read_text())
    records = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()]
    provenance = json.loads((DIRECTORY / "provenance.json").read_text())
    assert len(cards) == len({card["id"] for card in cards}) == 10
    assert len({card["oracle_id"] for card in cards}) == 10
    assert Counter(card["lang"] for card in cards) == Counter(dict.fromkeys(["fr", "de", "es", "pt", "it"], 2))
    assert len(records) == 20

    excluded_names = set(provenance["excluded_names_at_acquisition"])
    excluded_oracles = set(provenance["excluded_oracle_ids_at_acquisition"])
    for corpus in ["training/corpus/cards.json", "training/latin-corpus/cards.json"]:
        for card in json.loads((ROOT / corpus).read_text()):
            excluded_names.add(card["name"].casefold())
            excluded_oracles.add(card["oracle_id"])
    for card in cards:
        assert card["name"].casefold() not in excluded_names
        assert card["oracle_id"] not in excluded_oracles
        bucket = int(hashlib.sha256(card["name"].casefold().encode()).hexdigest()[:8], 16) % 10
        assert bucket == card["canonical_split_bucket"] == 0
        assert card["manually_reviewed"]
        assert card["manual_transcription"] == card["expected_text_nfc"] == unicodedata.normalize("NFC", card["printed_name"])
        assert any(ord(character) > 127 for character in card["manual_transcription"])
        left, top, right, bottom = card["crop_bounds_xyxy"]
        assert 0 <= left < right <= 672 and 0 <= top < bottom <= 936
        clean_path = DIRECTORY / "images" / card["clean"]
        blur_path = DIRECTORY / "images" / card["blur"]
        assert digest(clean_path.read_bytes()) == card["clean_sha256"]
        assert digest(blur_path.read_bytes()) == card["blur_sha256"]
        with Image.open(clean_path) as image:
            assert image.size == (right - left, bottom - top)
        with Image.open(blur_path) as image:
            assert image.size == (round((right - left) * .3), round((bottom - top) * .3))
        observations = [record for record in records if record["card_id"] == card["id"]]
        assert Counter(record["severity"] for record in observations) == {"clean": 1, "severe": 1}
        for record in observations:
            assert record["text"] == card["manual_transcription"] and record["split"] == "test"
            assert record["image_sha256"] == digest((DIRECTORY / record["path"]).read_bytes())
            assert record["language"] == card["lang"]
            assert record["oracle_id"] == card["oracle_id"] and record["card_name"] == card["name"]
            assert record["bbox"] == card["crop_bounds_xyxy"]
        if args.download_sources:
            request = urllib.request.Request(card["source_image_url"], headers={"User-Agent": "ocracle-accent-evaluation/0.1"})
            source = urllib.request.urlopen(request, timeout=90).read()
            assert digest(source) == card["source_image_sha256"]
            with Image.open(io.BytesIO(source)) as image:
                clean = image.convert("RGB").crop(card["crop_bounds_xyxy"])
            blur = clean.filter(ImageFilter.GaussianBlur(2.4)).resize(
                (round(clean.width * .3), round(clean.height * .3)), Image.Resampling.BILINEAR,
            )
            for image, expected in [(clean, card["clean_sha256"]), (blur, card["blur_sha256"])]:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                assert digest(buffer.getvalue()) == expected
    print("Verified 10 independent accented titles, 20 derivatives, split isolation and frozen provenance")


if __name__ == "__main__":
    main()
