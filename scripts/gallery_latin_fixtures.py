#!/usr/bin/env python3
"""Import frozen, visually checked Latin-accent title crops into the static gallery.

Copies exact image bytes from the independent test set without changing the
original English gallery fixtures or recorded model observations.
"""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training/latin-eval"
DESTINATION = ROOT / "docs/gallery/assets/latin"
LANGUAGES = {"fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese"}


def sha256(path):
    """Hash source data before copying a frozen artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Build a gallery manifest from the ten independently reviewed printings."""
    rows = [json.loads(line) for line in (SOURCE / "manifest.jsonl").read_text().splitlines()]
    cards = json.loads((SOURCE / "cards.json").read_text())
    DESTINATION.mkdir(parents=True, exist_ok=True)
    samples = []
    for card in cards:
        if not card["manually_reviewed"]:
            raise ValueError("Only visually reviewed printings can be imported")
        variants = []
        for row in [row for row in rows if row["card_id"] == card["id"]]:
            source = SOURCE / row["path"]
            if sha256(source) != row["image_sha256"]:
                raise ValueError(f"Independent test crop changed: {source}")
            filename = Path(row["path"]).name
            shutil.copyfile(source, DESTINATION / filename)
            variants.append({"id": "clean" if row["severity"] == "clean" else "blur",
                             "label": "Original scan crop" if row["severity"] == "clean" else "Severe blur",
                             "image": filename, "transform": row["transformation"]})
        if {variant["id"] for variant in variants} != {"clean", "blur"}:
            raise ValueError("Each imported printing needs both original and severe variants")
        samples.append({
            "id": f"latin-{card['identifier']}", "title": f"{LANGUAGES[card['lang']]} · accented title",
            "category": "title", "language": card["lang"],
            "font": f"Printed {card["set"].upper()} localized title; preserve visible diacritics",
            "ground_truth": card["manual_transcription"],
            "transcription": f"Manual visual transcription of this printed crop, reviewed {card['reviewed_on']}; not English Oracle text.",
            "crop_bounds_xyxy": card["crop_bounds_xyxy"],
            "source": {"name": card["name"], "print_id": card["id"], "oracle_id": card["oracle_id"],
                       "language": card["lang"], "set": card["set"].upper(), "collector_number": card["collector_number"],
                       "scryfall_url": card["scryfall_uri"], "image_url": card["source_image_url"],
                       "source_image_sha256": card["source_image_sha256"]},
            "variants": variants,
        })
    manifest = {"schema_version": 1, "held_out": True, "verified_on": "2026-10-19",
                "covered_languages": sorted({sample["language"] for sample in samples}),
                "language_note": "Fixed accented title examples from French, German, Spanish, Italian and Portuguese printings; ten titles do not establish complete language or font coverage.",
                "source_manifest_sha256": sha256(SOURCE / "manifest.jsonl"),
                "source_cards_sha256": sha256(SOURCE / "cards.json"),
                "source_documents": {"training/latin-eval/manifest.jsonl": sha256(SOURCE / "manifest.jsonl"),
                                     "training/latin-eval/cards.json": sha256(SOURCE / "cards.json")},
                "notes": "Independent test identities excluded from custom fine-tuning; unknown original pretrained model training membership. Selection did not filter on OCR output or confidence.",
                "samples": samples}
    (DESTINATION / "fixtures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Imported {len(samples)} fixed localized printings, {len(rows)} exact crop images")


if __name__ == "__main__":
    main()
