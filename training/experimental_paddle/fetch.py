#!/usr/bin/env python3
"""Download and verify pinned upstream model files and original fixture scans."""

from pathlib import Path
import argparse
import hashlib
import io
import json
import urllib.request

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
REVISION = "b8f84f0b80c529de40b4fbb3544b84fa7233a513"
FILES = {
    "inference.onnx": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    "inference.yml": "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
}


def fetch(url, destination, expected):
    """Keep cached bytes only when they have the requested immutable digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "ocracle research"})
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {destination}: {actual}")
    print(destination.name, actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scans", action="store_true")
    parser.add_argument("--additional", action="store_true", help="Fetch pinned v5 and Hebrew experiments too")
    args = parser.parse_args()
    for filename, expected in FILES.items():
        url = f"https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/{REVISION}/{filename}"
        fetch(url, ROOT / "cache" / filename, expected)
    if args.additional:
        for model in json.loads((ROOT / "additional-models.json").read_text())["models"]:
            for file in model["files"]:
                fetch(file["url"], ROOT / "cache" / file["cache_path"], file["sha256"])
    if args.source_scans:
        fixtures = json.loads((ROOT / "fixtures.json").read_text())
        for source in fixtures["sources"]:
            fetch(source["source_url"], ROOT / "cache" / (source["card_id"] + ".png"), source["source_sha256"])
        for case in fixtures["cases"]:
            source = Image.open(ROOT / "cache" / (case["card_id"] + ".png")).convert("RGB")
            crop = source.crop(case["bbox"])
            if case["severity"] == "severe":
                crop = crop.filter(ImageFilter.GaussianBlur(2.4)).resize(
                    (max(1, round(crop.width * .3)), max(1, round(crop.height * .3))),
                    Image.Resampling.BILINEAR,
                )
            output = io.BytesIO()
            crop.save(output, format="PNG")
            assert hashlib.sha256(output.getvalue()).hexdigest() == case["sha256"], case["id"]
            assert hashlib.sha256((ROOT / case["path"]).read_bytes()).hexdigest() == case["sha256"]
        print("All eight exact crop/degradation byte hashes reproduced from source scans.")


if __name__ == "__main__":
    main()
