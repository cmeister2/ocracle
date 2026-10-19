#!/usr/bin/env python3
"""Rebuild held-out, manually transcribed card crops for the static gallery.

Requires Pillow. Downloads are cached and Scryfall API requests are spaced out.
These exact print IDs and card names must remain outside MTG model training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/gallery/assets"
CACHE = ROOT / ".cache/gallery"
HEADERS = {
    "User-Agent": "ocracle-gallery/0.1 (+https://github.com/cmeister2/ocracle)",
    "Accept": "application/json",
}
PRINTS = {
    "lea-210": ("d4f1cc9e-4f99-4c26-ac1b-8ef069fa8ceb", "Llanowar Elves"),
    "8ed-59": ("3b15648c-1d88-42ec-b2f7-aab9a8c256a2", "Air Elemental"),
    "m15-52": ("ddd1187d-f43f-4508-b312-7af0bb8b9789", "Divination"),
    "m15-118": ("9f8723f6-96b1-4441-8a57-435fb82d6db4", "Typhoid Rats"),
    "fut-19": ("f6ccfe28-ec08-4751-b65c-c40b2bafa955", "Blade of the Sixth Pride"),
    "plc-25": ("7d48d622-f397-4f31-b1a5-0c23f60aa71c", "Mana Tithe"),
    "vow-331": ("1830002a-b512-4e51-b3fa-0137d2b4c749", "Thalia, Guardian of Thraben"),
    "mp2-12": ("a9b037f1-3298-4ba8-92a8-0843f6e497d7", "Daze"),
    "afr-350": ("c4c35234-2518-494c-8c1e-fd5702145967", "Cave of the Frost Dragon"),
}
# Coordinates and transcriptions refer to the downloaded 745 x 1040 print scans,
# not current Oracle wording. The scans were visually checked on 2026-10-19.
SAMPLES = [
    ("goudy-title", "lea-210", "Goudy Medieval · 1993 title", "title", "Goudy Medieval", [47, 47, 320, 94], "Llanowar Elves"),
    ("matrix-title", "8ed-59", "Matrix Bold · 2003 title", "title", "Matrix Bold", [59, 57, 326, 102], "Air Elemental"),
    ("beleren-title", "m15-52", "Beleren · 2014 title", "title", "Beleren", [59, 56, 260, 102], "Divination"),
    ("historic-rules", "lea-210", "Original printed rules", "rules", "Plantin family", [116, 635, 623, 673], "Tap to add 1 green mana to your"),
    ("modern-rules", "m15-52", "Modern printed rules", "rules", "Plantin family", [60, 717, 343, 759], "Draw two cards."),
    ("historic-flavor", "lea-210", "Historic italic flavor", "italic", "Plantin family italic", [116, 736, 640, 776], "Whenever the Llanowar Elves gather"),
    ("modern-flavor", "m15-52", "Modern italic flavor", "italic", "Plantin family italic", [62, 773, 626, 816], "“The key to unlocking this puzzle is"),
    ("italic-reminder", "8ed-59", "Italic reminder text", "italic", "Plantin family italic", [70, 746, 567, 785], "except by creatures with flying.)"),
    ("mixed-reminder", "m15-118", "Rules with italic reminder", "italic", "Plantin family regular + italic", [61, 662, 656, 704], "Deathtouch (Any amount of damage"),
    ("future-title", "fut-19", "Future-shifted title · 2007", "title", "Future Sight frame title treatment", [129, 60, 584, 106], "Blade of the Sixth Pride"),
    ("colorshift-title", "plc-25", "Colorshifted white title · 2007", "title", "Matrix Bold, white text with shadow", [59, 62, 279, 107], "Mana Tithe"),
    ("dracula-title", "vow-331", "Dracula showcase · inverted title", "title", "Dracula showcase title treatment", [56, 60, 291, 105], "Mina Harker"),
    ("invocation-title", "mp2-12", "Invocation display title · 2017", "title", "Amonkhet Invocation display; ornamental glyphs outside the word are excluded", [318, 59, 425, 98], "DAZE"),
    ("module-title", "afr-350", "D&D module showcase title · 2021", "title", "D&D module showcase display serif; second printed title line", [192, 102, 552, 148], "Frost Dragon"),
    ("collector-type", "m15-52", "Small collector information", "metadata", "Small sans-serif collector type; exact family not verified", [45, 969, 195, 990], "052/269 C"),
]


def fetch(url: str) -> bytes:
    """Download a single resource using a descriptive client identity."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response:
        result = response.read()
    time.sleep(0.15)
    return result


def main() -> None:
    """Download fixed prints and export repeatable crops plus blur variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh cached source scans and metadata")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = {}
    images = {}
    for key, (print_id, name) in PRINTS.items():
        metadata_path = CACHE / f"{key}.json"
        image_path = CACHE / f"{key}.png"
        if args.refresh or not metadata_path.exists():
            metadata_path.write_bytes(fetch(f"https://api.scryfall.com/cards/{print_id}"))
        metadata = json.loads(metadata_path.read_text())
        if metadata["id"] != print_id:
            raise ValueError(f"Unexpected print metadata for {key}")
        if args.refresh or not image_path.exists():
            image_path.write_bytes(fetch(metadata["image_uris"]["png"]))
        image_bytes = image_path.read_bytes()
        images[key] = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if images[key].size != (745, 1040):
            raise ValueError(f"Source image dimensions changed for {key}; review crop coordinates")
        sources[key] = {
            "name": name,
            "print_id": print_id,
            "set": metadata["set_name"],
            "language": metadata["lang"],
            "collector_number": metadata["collector_number"],
            "scryfall_url": metadata["scryfall_uri"],
            "image_url": metadata["image_uris"]["png"],
            "source_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "artist": metadata["artist"],
        }
    samples = []
    for sample_id, print_key, title, category, font, bounds, truth in SAMPLES:
        crop = images[print_key].crop(bounds)
        clean_name = f"{sample_id}.png"
        crop.save(ASSETS / clean_name, optimize=True)
        # A fixed stress condition, deliberately harder than ordinary scanning.
        # Blur applies before downsampling to 30%, so the final image loses detail.
        degraded = crop.filter(ImageFilter.GaussianBlur(radius=2.4)).resize(
            (max(1, round(crop.width * 0.3)), max(1, round(crop.height * 0.3))), Image.Resampling.BILINEAR
        )
        blur_name = f"{sample_id}-blur.png"
        degraded.save(ASSETS / blur_name, optimize=True)
        samples.append({
            "id": sample_id,
            "title": title,
            "category": category,
            "language": sources[print_key]["language"],
            "font": font,
            "ground_truth": truth,
            "transcription": "Manual visual transcription of this printed crop; not Oracle text.",
            "crop_bounds_xyxy": bounds,
            "source": sources[print_key],
            "variants": [
                {"id": "clean", "label": "Original scan crop", "image": clean_name, "transform": "Crop only"},
                {"id": "blur", "label": "Severe blur", "image": blur_name, "transform": "Gaussian σ = 2.4 source pixels; downsample to 30% with bilinear resampling"},
            ],
        })
    manifest = {
        "schema_version": 1,
        "held_out": True,
        "covered_languages": ["en"],
        "language_note": "These current fixtures and recognition checkpoints cover English only. The project objective includes every printed MTG language; other scripts remain untrained and unverified.",
        "verified_on": "2026-10-19",
        "notes": "All card names and source print IDs in this manifest are reserved for gallery evaluation and excluded from custom training. Font labels identify the standard card-frame families; this small sample does not establish coverage of every MTG treatment or language.",
        "font_sources": [
            {"title": "Wizards: Frames of Reference (2003)", "url": "https://magic.wizards.com/en/news/making-magic/frames-reference-2003-01-27"},
            {"title": "Wizards: From the Director’s Chair: 2013 (Matrix and Beleren)", "url": "https://magic.wizards.com/en/news/making-magic/directors-chair-2013-2014-01-06"},
            {"title": "Wizards: Dominaria Frame, Template, and Rules Changes (italics)", "url": "https://magic.wizards.com/en/news/announcements/dominaria-frame-template-and-rules-changes-2018-03-21"},
            {"title": "Wizards: The Future Is Now, Part I (future-shifted frame)", "url": "https://magic.wizards.com/en/news/making-magic/future-now-part-i-2007-04-09"},
            {"title": "Wizards: Masterpiece Series: Amonkhet Invocations", "url": "https://magic.wizards.com/en/news/card-preview/masterpiece-series-amonkhet-invocations-2017-03-29"},
            {"title": "Wizards: Crimson Vow Booster Fun (Dracula)", "url": "https://magic.wizards.com/en/news/feature/sink-your-teeth-booster-fun-innistrad-crimson-vow-2021-10-28"},
            {"title": "Wizards: Chaos Theory (colorshifted white text)", "url": "https://magic.wizards.com/en/news/making-magic/chaos-theory-2007-01-08"},
            {"title": "Wizards: Behind the Scenes of Booster Fun for Adventures in the Forgotten Realms (module treatment)", "url": "https://magic.wizards.com/en/news/feature/behind-scenes-booster-fun-adventures-forgotten-realms-2021-06-29"},
        ],
        "samples": samples,
    }
    (ASSETS / "fixtures.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(samples)} held-out samples, each with original and severe blur crops")


if __name__ == "__main__":
    main()
