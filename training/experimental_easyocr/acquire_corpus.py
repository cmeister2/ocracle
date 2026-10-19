#!/usr/bin/env python3
"""Freeze a small localized MTG alphabet-audit corpus from Scryfall bulk data.

No API crawl is needed. Localized printing fields are preserved and never
replaced with current English Oracle text. This audits labels, not glyph accuracy.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BULK_NAME = "all-cards-20261019091831.jsonl.gz"
BULK_SHA256 = "b608540096ca0eead69aaa7d9b2302d480872f2a091dfd5bfafc0128bf69ea8f"
COHORTS = ["4bb", "5ed", "7ed", "8ed", "fut", "m15", "j25", "neo", "sta", "khm", "vow"]
EVALUATION_IDS = {"2607d870-4995-489f-b5f9-76999a45958d", "b024b2e5-4c29-403d-a287-25b48bbe82b6"}
FIELDS = ["id", "oracle_id", "name", "lang", "set", "set_name", "collector_number", "released_at", "layout",
          "printed_name", "printed_text", "flavor_text", "image_status", "highres_image", "scryfall_uri",
          "image_uris", "frame", "frame_effects", "card_faces", "artist"]


def main():
    """Select reproducible Japanese printing samples across historic set cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk", type=Path, default=ROOT / "training/data/scryfall" / BULK_NAME)
    parser.add_argument("--per-cohort", type=int, default=8)
    parser.add_argument("--output", type=Path, default=HERE / "corpus/cards.json")
    args = parser.parse_args()
    with args.bulk.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != BULK_SHA256:
        raise ValueError("Unexpected Scryfall bulk snapshot; record and review a new digest before changing the corpus")
    candidates = {cohort: [] for cohort in COHORTS}
    held_out = []
    with gzip.open(args.bulk, "rt") as source:
        for line in source:
            card = json.loads(line)
            if card["lang"] != "ja":
                continue
            if card["id"] in EVALUATION_IDS:
                held_out.append({key: card[key] for key in FIELDS if key in card})
            if card["set"] not in candidates or not (card.get("printed_name") or card.get("card_faces")):
                continue
            stable_order = hashlib.sha256(("ocracle-ja-alphabet-v1:" + card["id"]).encode()).hexdigest()
            group = candidates[card["set"]]
            group.append((stable_order, {key: card[key] for key in FIELDS if key in card}))
            group.sort(key=lambda entry: entry[0])
            del group[args.per_cohort:]
    cards = {card["id"]: card for card in held_out}
    for group in candidates.values():
        cards.update((card["id"], card) for _, card in group)
    result = {
        "schema_version": 1,
        "source": "Scryfall all-cards bulk snapshot",
        "source_url": "https://scryfall.com/docs/api/bulk-data",
        "bulk_filename": BULK_NAME,
        "bulk_sha256": BULK_SHA256,
        "selection": f"Japanese printed fields; {args.per_cohort} deterministic hash-ranked printings per historic cohort, plus fixed actual-image evaluation printings",
        "cohorts": {cohort: len(group) for cohort, group in candidates.items()},
        "evaluation_print_ids": sorted(EVALUATION_IDS),
        "notes": "An alphabet audit of actual localized print labels. Most label strings have not been visually transcribed; image-level evaluation uses a separate manually checked fixture manifest. This is not a training or recognition-accuracy result.",
        "cards": sorted(cards.values(), key=lambda card: (card["set"], card["collector_number"], card["id"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(cards)} localized printings from {len(candidates)} cohorts: {args.output}")


if __name__ == "__main__":
    main()
