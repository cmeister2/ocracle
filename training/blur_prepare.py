#!/usr/bin/env python3
"""Reproduce the manually reviewed, development-only printed-line blur corpus.

No recognizer is imported or invoked. Source/annotation digests are immutable
inputs; changed network imagery fails closed instead of replacing evidence.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import unicodedata
import urllib.request

if not __debug__:
    raise RuntimeError("Integrity checks require non-optimized Python; do not use -O or PYTHONOPTIMIZE")

import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "training/blur-corpus"
CONDITIONS = [
    ("clean", {"operation": "identity"}),
    ("gaussian-1p2", {"operation": "gaussian", "sigma_native_pixels": 1.2}),
    ("gaussian-2p4", {"operation": "gaussian", "sigma_native_pixels": 2.4}),
    ("downsample-50pct", {"operation": "downsample", "scale": .5, "resampling": "bilinear", "rounding": "half_up", "upscale": False}),
    ("motion-9px", {"operation": "horizontal_box_motion", "length_native_pixels": 9, "edge": "replicate", "rounding": "half_up"}),
    ("jpeg-q25", {"operation": "jpeg", "quality": 25, "subsampling": 2, "optimize": False, "progressive": False}),
    ("severe", {"operation": "gaussian_then_downsample", "sigma_native_pixels": 2.4, "scale": .3, "resampling": "bilinear", "rounding": "half_up", "upscale": False}),
]


def digest(data):
    """Return the SHA256 of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def read_json(name):
    return json.loads((DIRECTORY / name).read_text())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def environment():
    return {"Pillow": PIL.__version__, "jpeg": features.version("jpg"), "zlib": features.version("zlib"), "webp": features.version("webp")}


def identities(value):
    """Read canonical identities from corpus and gallery metadata structures."""
    names, oracles = set(), set()
    if isinstance(value, dict):
        if value.get("oracle_id"):
            oracles.add(value["oracle_id"])
        if value.get("card_name"):
            names.add(value["card_name"].casefold())
        if value.get("name") and any(key in value for key in ["oracle_id", "lang", "print_id", "printed_name"]):
            names.add(value["name"].casefold())
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        children = []
    for child in children:
        if isinstance(child, (dict, list)):
            child_names, child_oracles = identities(child)
            names.update(child_names)
            oracles.update(child_oracles)
    return names, oracles


def validate_inputs(cards, annotations, provenance):
    assert digest((DIRECTORY / "cards.json").read_bytes()) == provenance["cards_sha256"]
    assert digest((DIRECTORY / "annotations.json").read_bytes()) == provenance["annotations_sha256"]
    assert len(cards) == len({card["id"] for card in cards}) == 16
    assert len({card["oracle_id"] for card in cards}) == 16
    assert len(annotations) == len({a["base_id"] for a in annotations}) == 64
    assert Counter(a["card_id"] for a in annotations) == dict.fromkeys([c["id"] for c in cards], 4)
    assert all(a["review_status"] == "pixel_reviewed" for a in annotations)
    excluded_names = set(provenance["identity_exclusions"]["canonical_names_casefold"])
    excluded_oracles = set(provenance["identity_exclusions"]["oracle_ids"])
    # Check the acquisition snapshot and all current versions of the same
    # corpora; later training must not silently absorb this development set.
    for source in provenance["identity_exclusions"]["sources"]:
        path = ROOT / source["path"]
        assert path.is_file(), f"Missing exclusion source: {path}"
        names, oracles = identities(json.loads(path.read_text()))
        excluded_names.update(names)
        excluded_oracles.update(oracles)
    for card in cards:
        assert card["name"].casefold() not in excluded_names, card["name"]
        assert card["oracle_id"] not in excluded_oracles, card["name"]
        assert card["review_status"] == "physical_language_verified"
        assert card["observed_language"] == card["lang"]
    by_id = {card["id"]: card for card in cards}
    for annotation in annotations:
        assert annotation["text"] == unicodedata.normalize("NFC", annotation["text"])
        assert annotation["text"] and annotation["text"] == annotation["text"].strip()
        assert annotation["style"] in ["title", "rules", "flavor", "reminder"]
        assert annotation["italic"] == (annotation["style"] in ["flavor", "reminder"])
        left, top, right, bottom = annotation["bbox"]
        width, height = by_id[annotation["card_id"]]["source_image_size"]
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
    assert provenance["partition"] == "development"
    assert provenance["model_inference_used_for_selection_or_labels"] is False
    assert provenance["conditions"] == [{"condition": name, **parameters} for name, parameters in CONDITIONS]
    for key in ["source_rejections", "vocabulary_audit"]:
        evidence = provenance[key]
        assert digest((DIRECTORY / evidence["path"]).read_bytes()) == evidence["sha256"]
    rejected = read_json(provenance["source_rejections"]["path"])
    assert len(rejected) == provenance["source_rejections"]["count"] == 3
    assert {c["id"] for c in cards}.isdisjoint({c["id"] for c in rejected})
    assert read_json(provenance["vocabulary_audit"]["path"])["annotations_sha256"] == provenance["annotations_sha256"]


def source_image(card, download):
    """Read the exact cached source, optionally acquiring missing bytes."""
    path = DIRECTORY / "cache" / f"{card['id']}.img"
    if not path.exists():
        assert download, f"Source absent: {path}; use --download-sources"
        request = urllib.request.Request(card["source_image_url"], headers={
            "User-Agent": "ocracle-blur-development/0.1 (https://github.com/cmeister2/ocracle)",
            "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.5",
        })
        with urllib.request.urlopen(request, timeout=90) as response:
            data = response.read()
        assert digest(data) == card["source_image_sha256"], f"Changed source: {card['name']}"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        time.sleep(.15)
    data = path.read_bytes()
    assert digest(data) == card["source_image_sha256"], f"Source hash mismatch: {path}"
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == card["source_image_format"]
        image = image.convert("RGB")
    assert list(image.size) == card["source_image_size"]
    assert digest(image.tobytes()) == card["source_pixel_sha256"]
    return image


def motion_blur(image, length=9):
    """Horizontal box convolution, edge repetition, integer half-up rounding."""
    width, height = image.size
    source = image.tobytes()
    result = bytearray(len(source))
    radius = length // 2
    for y in range(height):
        for channel in range(3):
            row = [source[(y * width + x) * 3 + channel] for x in range(width)]
            padded = [row[0]] * radius + row + [row[-1]] * radius
            total = sum(padded[:length])
            for x in range(width):
                result[(y * width + x) * 3 + channel] = (total + radius) // length
                if x + 1 < width:
                    total += padded[x + length] - padded[x]
    return Image.frombytes("RGB", image.size, bytes(result))


def artifact(image, condition):
    """Encode one deterministic derivative of a native clean line."""
    if condition == "gaussian-1p2":
        image = image.filter(ImageFilter.GaussianBlur(1.2))
    elif condition == "gaussian-2p4":
        image = image.filter(ImageFilter.GaussianBlur(2.4))
    elif condition == "downsample-50pct":
        image = image.resize(((image.width + 1) // 2, (image.height + 1) // 2), Image.Resampling.BILINEAR)
    elif condition == "motion-9px":
        image = motion_blur(image)
    elif condition == "severe":
        image = image.filter(ImageFilter.GaussianBlur(2.4)).resize(
            ((image.width * 3 + 5) // 10, (image.height * 3 + 5) // 10), Image.Resampling.BILINEAR,
        )
    stream = io.BytesIO()
    extension = "jpg" if condition == "jpeg-q25" else "png"
    if extension == "jpg":
        image.save(stream, format="JPEG", quality=25, subsampling=2, optimize=False, progressive=False)
    else:
        image.save(stream, format="PNG", compress_level=6, optimize=False)
    return stream.getvalue(), extension


def manifest_record(card, annotation, condition, parameters, data, extension, clean):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
    return {
        "base_id": annotation["base_id"],
        "path": f"images/{annotation['base_id']}/{condition}.{extension}",
        "text": annotation["text"], "language": card["lang"],
        "card_id": card["id"], "oracle_id": card["oracle_id"], "card_name": card["name"],
        "set": card["set"], "collector_number": card["collector_number"], "frame": card["frame"],
        "split": "development", "style": annotation["style"], "italic": annotation["italic"],
        "condition": condition, "transform": parameters,
        "bbox": annotation["bbox"], "bbox_convention": "native source pixels: left,top,right,bottom; exclusive right/bottom",
        "image_url": card["source_image_url"], "source_image_sha256": card["source_image_sha256"],
        "source_pixel_sha256": card["source_pixel_sha256"], "clean_pixel_sha256": digest(clean.tobytes()),
        "image_sha256": digest(data), "image_pixel_sha256": digest(image.tobytes()), "image_size": list(image.size),
    }


def review_images(cards, annotations, images):
    """Make human evidence sheets without any recognizer output."""
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15)
    outputs = {}
    for start in range(0, len(annotations), 8):
        out = Image.new("RGB", (810, 900), "#eeeeee")
        draw, y = ImageDraw.Draw(out), 8
        for annotation in annotations[start:start + 8]:
            crop = images[annotation["card_id"]].crop(annotation["bbox"])
            draw.text((10, y), f"{annotation['base_id']} · {annotation['style']} · {annotation['bbox']}", font=font, fill="black")
            y += 21
            draw.text((10, y), annotation["text"], font=font, fill="black")
            y += 24
            out.paste(crop, (10, y))
            y += crop.height + 14
        outputs[f"review/clean-{start // 8 + 1:02d}.png"] = out.crop((0, 0, 810, y))
    # Full-card thumbnails demonstrate physical printed language and treatment.
    out = Image.new("RGB", (1120, 1680), "white")
    draw = ImageDraw.Draw(out)
    for index, card in enumerate(cards):
        x, y = index % 4 * 280, index // 4 * 420
        image = images[card["id"]]
        thumb = image.resize((260, round(image.height * 260 / image.width)), Image.Resampling.BILINEAR)
        draw.text((x + 5, y + 3), f"{card['lang']} · {card['set']} · {card['collector_number']}", font=font, fill="black")
        out.paste(thumb, (x + 5, y + 26))
    outputs["review/sources.png"] = out
    # One real italic line, every predeclared condition; downsampling is shown
    # at native output size, with no implicit upscaling of the hard examples.
    annotation = next(a for a in annotations if a["base_id"] == "de-dom-88-03")
    clean = images[annotation["card_id"]].crop(annotation["bbox"])
    out = Image.new("RGB", (810, 620), "#eeeeee")
    draw, y = ImageDraw.Draw(out), 10
    for condition, parameters in CONDITIONS:
        data, _ = artifact(clean, condition)
        draw.text((10, y), condition + " · " + json.dumps(parameters, ensure_ascii=False), font=font, fill="black")
        y += 23
        with Image.open(io.BytesIO(data)) as image:
            out.paste(image.convert("RGB"), (10, y))
            y += image.height + 16
    outputs["review/degradations.png"] = out.crop((0, 0, 810, y))
    for path, image in outputs.items():
        stream = io.BytesIO()
        image.save(stream, format="PNG", compress_level=6, optimize=False)
        (DIRECTORY / path).parent.mkdir(exist_ok=True)
        (DIRECTORY / path).write_bytes(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate frozen evidence; no network unless explicitly requested")
    parser.add_argument("--download-sources", action="store_true", help="Fetch missing scans, accepting only frozen source digests")
    parser.add_argument("--rebuild", action="store_true", help="Reproduce derivatives/review sheets from frozen annotations")
    args = parser.parse_args()
    assert args.check or args.rebuild, "Choose --check or --rebuild"
    cards, annotations, provenance = read_json("cards.json"), read_json("annotations.json"), read_json("provenance.json")
    validate_inputs(cards, annotations, provenance)
    records = [json.loads(line) for line in (DIRECTORY / "manifest.jsonl").read_text().splitlines()] if args.check else []
    if args.check:
        assert digest((DIRECTORY / "manifest.jsonl").read_bytes()) == provenance["manifest_sha256"]
        assert len(records) == 448
        assert Counter(r["condition"] for r in records) == dict.fromkeys([name for name, _ in CONDITIONS], 64)
        assert len({r["path"] for r in records}) == len(records)
        expected_cases = {(a["base_id"], condition) for a in annotations for condition, _ in CONDITIONS}
        assert len({(r["base_id"], r["condition"]) for r in records}) == len(records)
        assert {(r["base_id"], r["condition"]) for r in records} == expected_cases
        by_annotation = {a["base_id"]: a for a in annotations}
        by_card = {c["id"]: c for c in cards}
        clean_pixels = {}
        for record in records:
            if record["condition"] == "clean":
                with Image.open(DIRECTORY / record["path"]) as image:
                    clean_pixels[record["base_id"]] = digest(image.convert("RGB").tobytes())
        for record in records:
            extension = "jpg" if record["condition"] == "jpeg-q25" else "png"
            assert record["path"] == f"images/{record['base_id']}/{record['condition']}.{extension}"
            annotation = by_annotation[record["base_id"]]
            card = by_card[annotation["card_id"]]
            assert record["text"] == annotation["text"]
            assert record["bbox"] == annotation["bbox"]
            assert record["style"] == annotation["style"]
            assert record["italic"] == annotation["italic"]
            assert record["card_id"] == card["id"]
            assert record["oracle_id"] == card["oracle_id"]
            assert record["card_name"] == card["name"]
            assert record["source_image_sha256"] == card["source_image_sha256"]
            assert record["source_pixel_sha256"] == card["source_pixel_sha256"]
            assert record["clean_pixel_sha256"] == clean_pixels[record["base_id"]]
            assert record["image_url"] == card["source_image_url"]
            assert record["set"] == card["set"]
            assert record["collector_number"] == card["collector_number"]
            assert record["frame"] == card["frame"]
            assert record["language"] == card["lang"]
            assert record["split"] == "development"
            assert record["transform"] == dict(CONDITIONS)[record["condition"]]
            data = (DIRECTORY / record["path"]).read_bytes()
            assert digest(data) == record["image_sha256"], record["path"]
            with Image.open(io.BytesIO(data)) as image:
                assert list(image.size) == record["image_size"]
                # RGB pixels are versioned too, but a different local JPEG
                # decoder need not invalidate exact preserved artifact bytes.
                if environment() == provenance["render_environment"]:
                    assert digest(image.convert("RGB").tobytes()) == record["image_pixel_sha256"]
        for evidence in provenance["review_evidence"]:
            assert digest((DIRECTORY / evidence["path"]).read_bytes()) == evidence["sha256"]
    available = all((DIRECTORY / "cache" / f"{c['id']}.img").exists() for c in cards)
    if args.rebuild or args.download_sources or available:
        assert environment() == provenance["render_environment"], f"Reproduction needs {provenance['render_environment']}; found {environment()}"
        images = {card["id"]: source_image(card, args.download_sources) for card in cards}
        by_id = {card["id"]: card for card in cards}
        generated, artifacts = [], []
        for annotation in annotations:
            card = by_id[annotation["card_id"]]
            clean = images[card["id"]].crop(annotation["bbox"])
            for condition, parameters in CONDITIONS:
                data, extension = artifact(clean, condition)
                record = manifest_record(card, annotation, condition, parameters, data, extension, clean)
                generated.append(record)
                if args.rebuild:
                    artifacts.append((record["path"], data))
        data = b"".join((json.dumps(record, ensure_ascii=False) + "\n").encode() for record in generated)
        assert digest(data) == provenance["manifest_sha256"], "Rendered manifest differs from frozen evidence"
        if args.check:
            assert generated == records
        if args.rebuild:
            for relative, artifact_data in artifacts:
                path = DIRECTORY / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(artifact_data)
            (DIRECTORY / "manifest.jsonl").write_bytes(data)
            review_images(cards, annotations, images)
    print("Verified 16 fresh canonical identities, 64 reviewed native lines, 448 development derivatives; no OCR used")


if __name__ == "__main__":
    main()
