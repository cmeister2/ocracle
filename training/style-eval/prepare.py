#!/usr/bin/env python3
"""Reproduce or verify the complete five-card poster-title challenge offline."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import unicodedata

from PIL import Image, ImageDraw, ImageFilter, ImageFont, __version__ as PILLOW_VERSION

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
SCANS = ROOT / "training/data/style-eval"
CORPORA = ["training/corpus/cards.json", "training/latin-corpus/cards.json", "training/latin-v2-corpus/cards.json"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_isolation(annotations):
    """Reject use of a challenge card identity in any current custom corpus."""
    names = {row["card_name"].casefold() for row in annotations}
    oracles = {row["oracle_id"] for row in annotations}
    prints = {row["card_id"] for row in annotations}
    counts = {}
    for relative in CORPORA:
        path = ROOT / relative
        rows = json.loads(path.read_text())
        for row in rows:
            if row.get("name", "").casefold() in names or row.get("oracle_id") in oracles or row.get("id") in prints:
                raise ValueError("Challenge identity occurs in custom training corpus: " + relative)
        counts[relative] = {"cards": len(rows), "sha256": digest(path)}
    # Check actually materialized manifests as well as metadata selections.
    for relative in ["training/data/manifest.jsonl", "training/data/latin/manifest.jsonl", "training/data/latin-v2/manifest.jsonl"]:
        path = ROOT / relative
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("card_name", "").casefold() in names or row.get("oracle_id") in oracles or row.get("card_id") in prints:
                raise ValueError("Challenge identity occurs in materialized custom training: " + relative)
    return counts


def review_image(records):
    """Show every frozen field and an enlarged severe counterpart for review."""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    height = sum(row["dimensions"][1] + 48 for row in records if row["severity"] == "clean")
    output = Image.new("RGB", (1380, height), "white")
    draw = ImageDraw.Draw(output)
    top = 0
    for index in range(0, len(records), 2):
        clean, severe = records[index:index + 2]
        draw.text((8, top), clean["text"] + " | original field / severe enlarged for display", font=font, fill="black")
        top += 30
        original = Image.open(DIRECTORY / clean["path"]).convert("RGB")
        blurred = Image.open(DIRECTORY / severe["path"]).convert("RGB")
        output.paste(original, (8, top))
        output.paste(blurred.resize(original.size, Image.Resampling.NEAREST), (700, top))
        top += original.height + 18
    encoded = io.BytesIO()
    output.save(encoded, format="PNG")
    return encoded.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify frozen artifacts without rewriting them")
    parser.add_argument("--download-sources", action="store_true", help="Fetch missing original scans and verify their frozen digests")
    parser.add_argument("--source-directory", type=Path, default=SCANS, help="Original scan cache; may be an empty temporary directory for independent replay")
    args = parser.parse_args()
    scans = args.source_directory
    if PILLOW_VERSION != "13.0.0":
        raise ValueError("Exact image reproduction requires Pillow13.0.0 (training/requirements.txt)")
    cards = json.loads((DIRECTORY / "cards.json").read_text())
    annotations = json.loads((DIRECTORY / "annotations.json").read_text())
    if (DIRECTORY / "provenance.json").exists():
        frozen = json.loads((DIRECTORY / "provenance.json").read_text())
        if frozen["cards_sha256"] != digest(DIRECTORY / "cards.json") or frozen["annotations_sha256"] != digest(DIRECTORY / "annotations.json"):
            raise ValueError("Frozen cases/transcriptions changed; preserve this benchmark and create a separately versioned challenge")
    if len(cards) != 5 or len(annotations) != 5 or {x["metadata"]["id"] for x in cards} != {x["card_id"] for x in annotations}:
        raise ValueError("The closed five-card publisher set must remain complete")
    corpora = verify_isolation(annotations)
    sources = {row["metadata"]["id"]: row for row in cards}
    records = []
    for annotation in annotations:
        source = sources[annotation["card_id"]]
        metadata = source["metadata"]
        if annotation["oracle_id"] != metadata["oracle_id"] or annotation["card_name"] != metadata["name"] or metadata["lang"] != "en":
            raise ValueError("Source identity/language differs from annotation")
        if annotation["text"] != unicodedata.normalize("NFC", annotation["text"]):
            raise ValueError("Challenge title is not NFC")
        # This is solely an identity sanity check. Scoring retains actual case
        # and both visible middle dots; it never substitutes canonical names.
        if annotation["text"].replace("·", " ").casefold() != metadata["name"].casefold():
            raise ValueError("Manual lettering does not identify the declared source card")
        scan = scans / (annotation["card_id"] + ".jpg")
        if not scan.exists() and args.download_sources:
            request = urllib.request.Request(source["image_url"], headers={"User-Agent": "ocracle-style-evaluation/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["scan_sha256"]:
                raise ValueError("Downloaded source digest differs: " + annotation["id"])
            scan.parent.mkdir(parents=True, exist_ok=True)
            scan.write_bytes(data)
        if scan.exists():
            if digest(scan) != source["scan_sha256"]:
                raise ValueError("Original source scan changed: " + annotation["id"])
            image = Image.open(scan).convert("RGB")
            if list(image.size) != source["image_size"]:
                raise ValueError("Original source dimensions changed")
            left, top, right, bottom = annotation["box"]
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("Title rectangle outside original source")
            clean = image.crop(annotation["box"])
        elif not args.check:
            raise ValueError("Source missing; use --download-sources")
        else:
            clean = None
        for severity in ["clean", "severe"]:
            relative = "images/" + annotation["id"] + "-" + severity + ".png"
            target = DIRECTORY / relative
            if clean is not None:
                crop = clean if severity == "clean" else clean.filter(ImageFilter.GaussianBlur(2.4)).resize(
                    (max(1, round(clean.width * .3)), max(1, round(clean.height * .3))), Image.Resampling.BILINEAR
                )
                output = io.BytesIO()
                crop.save(output, format="PNG")
                encoded = output.getvalue()
                if args.check:
                    if not target.exists() or target.read_bytes() != encoded:
                        raise ValueError("Frozen crop does not reproduce from source: " + relative)
                else:
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(encoded)
            if not target.exists():
                raise ValueError("Frozen crop missing: " + relative)
            with Image.open(target) as result:
                size = list(result.size)
            records.append({**annotation, "path": relative, "severity": severity, "split": "external-frozen-challenge", "source": "reviewed-real-scan", "image_sha256": digest(target), "dimensions": size, "image_url": source["image_url"], "scan_sha256": source["scan_sha256"], "source_size": source["image_size"], "publisher_source": source["publisher_source"]})
    manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    path = DIRECTORY / "manifest.jsonl"
    if args.check:
        if path.read_text() != manifest:
            raise ValueError("Frozen challenge manifest differs")
        provenance = json.loads((DIRECTORY / "provenance.json").read_text())
        if provenance["manifest_sha256"] != hashlib.sha256(manifest.encode()).hexdigest() or provenance["cards_sha256"] != digest(DIRECTORY / "cards.json") or provenance["annotations_sha256"] != digest(DIRECTORY / "annotations.json"):
            raise ValueError("Frozen challenge provenance differs")
        if digest(DIRECTORY / "review.png") != provenance["review_sha256"]:
            raise ValueError("Frozen visual-review contact sheet changed")
    else:
        path.write_text(manifest)
        (DIRECTORY / "review.png").write_bytes(review_image(records))
        provenance = {"schema_version": 1, "frozen_on": "2026-10-19", "cards": 5, "observations": 10, "manifest_sha256": digest(path), "cards_sha256": digest(DIRECTORY / "cards.json"), "annotations_sha256": digest(DIRECTORY / "annotations.json"), "review_sha256": digest(DIRECTORY / "review.png"), "publisher_set": "Our Show Is on Friday, Can You Make It? (2021 Secret Lair)", "selection": "All five cards enumerated in the publisher drop; all complete title fields manually fixed before any OCR runs. No OCR filtering or training.", "normalization": "NFC plus collapsed whitespace only; case, apostrophe and visible middle-dot separators are scored literally.", "degradation": {"gaussian_sigma_source_pixels": 2.4, "linear_scale": .3, "resample": "Pillow BILINEAR", "pillow_version": PILLOW_VERSION}, "frozen_custom_training_exclusion_check": corpora, "training_isolation": "No name, Oracle ID or print ID appears in current English, Latin V1 or Latin V2 corpus selections/materialized manifests. Upstream pretrained-model membership is unknown.", "scope": "Hand-drawn Latin title lettering and its actual two-dimensional layout; rectangular crops retain incidental source pixels. Not a pure single-font classification benchmark.", "rights": "Original card imagery/lettering belongs to Wizards of the Coast and credited artists; frozen crop evidence is research data, not relicensed as MIT source code."}
        (DIRECTORY / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cards": 5, "observations": len(records), "manifest_sha256": digest(path), "source_scans_available": sum((scans / (a['card_id'] + '.jpg')).exists() for a in annotations), "training_identity_overlap": False}))


if __name__ == "__main__":
    main()
