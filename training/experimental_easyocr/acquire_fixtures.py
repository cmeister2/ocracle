#!/usr/bin/env python3
"""Recreate manually checked Japanese print crops with severe-blur stress pairs."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
PRINTS = {
    "2607d870-4995-489f-b5f9-76999a45958d": {"cache": "ja-opt.png", "sha256": "ee8a376e6e09933dbef45eb32532cafd76f9cef6645ff1eb03cd1f4bb883ac24"},
    "b024b2e5-4c29-403d-a287-25b48bbe82b6": {"cache": "m15-52-ja.png", "sha256": "ecdf738be7a49dfbe3d29e8060498232b262aa72fc4a822cfafa1438b5d593d4"},
}
SAMPLES = [
    ("ja-title", "2607d870-4995-489f-b5f9-76999a45958d", "title", [59, 72, 131, 104], "選択"),
    ("ja-furigana", "2607d870-4995-489f-b5f9-76999a45958d", "furigana", [60, 55, 124, 73], "せんたく"),
    ("ja-rules", "2607d870-4995-489f-b5f9-76999a45958d", "rules", [61, 787, 322, 829], "カード１枚を引く。"),
    ("ja-reminder", "2607d870-4995-489f-b5f9-76999a45958d", "rules and reminder", [61, 681, 675, 715], "占術１を行う。（あなたのライブラリーの一番上に"),
    ("ja-flavor", "2607d870-4995-489f-b5f9-76999a45958d", "flavor", [61, 851, 673, 882], "「風雲船長の命令は予想がつくな。『危険な方の航路を"),
    ("ja-lowres-title", "b024b2e5-4c29-403d-a287-25b48bbe82b6", "title in low-resolution source", [59, 56, 181, 102], "予言"),
    ("ja-lowres-rules", "b024b2e5-4c29-403d-a287-25b48bbe82b6", "rules in low-resolution source", [61, 734, 329, 777], "カードを２枚引く。"),
]


def main():
    """Acquire pinned source images and apply fixed reviewed crop coordinates."""
    cards = {card["id"]: card for card in json.loads((HERE / "corpus/cards.json").read_text())["cards"]}
    cache = HERE / "cache/images"
    output = HERE / "fixtures"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for print_id, entry in PRINTS.items():
        card = cards[print_id]
        path = cache / entry["cache"]
        if not path.exists():
            request = urllib.request.Request(card["image_uris"]["png"], headers={
                "User-Agent": "ocracle-ja-fixtures/0.1 (+https://github.com/cmeister2/ocracle)", "Accept": "image/png"})
            with urllib.request.urlopen(request, timeout=90) as response:
                path.write_bytes(response.read())
            time.sleep(0.15)
        image = Image.open(path).convert("RGB")
        if image.size != (745, 1040):
            raise ValueError("Source dimensions changed; re-review all crop coordinates")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Source pixels changed for {print_id}; re-review the source before replacing its digest")
        sources[print_id] = {
            "print_id": print_id, "card_name": card["name"], "printed_name": card["printed_name"],
            "language": card["lang"], "set": card["set"], "collector_number": card["collector_number"],
            "scryfall_url": card["scryfall_uri"], "image_url": card["image_uris"]["png"],
            "image_status": card["image_status"], "source_image_sha256": actual,
            "manual_image_review": "2026-10-19: image visually verified to contain Japanese text; not an English fallback",
        }
    samples = []
    for sample_id, print_id, treatment, bounds, truth in SAMPLES:
        image = Image.open(cache / PRINTS[print_id]["cache"]).convert("RGB").crop(bounds)
        clean = f"{sample_id}.png"
        image.save(output / clean, optimize=True)
        degraded = image.filter(ImageFilter.GaussianBlur(2.4)).resize(
            (max(1, round(image.width * .3)), max(1, round(image.height * .3))), Image.Resampling.BILINEAR)
        blur = f"{sample_id}-blur.png"
        degraded.save(output / blur, optimize=True)
        samples.append({
            "id": sample_id, "language": "ja", "treatment": treatment, "ground_truth": truth,
            "reference": "Manual transcription of this printed crop, preserving fullwidth digits and punctuation",
            "typography_note": "Japanese flavor/reminder glyphs are upright in this print; this does not establish Latin italic or vertical-writing support",
            "crop_bounds_xyxy": bounds, "source": sources[print_id],
            "variants": [{"id": "source", "image": clean, "transform": "Crop only; see source image_status for intrinsic blur"},
                         {"id": "blur", "image": blur, "transform": "Gaussian sigma2.4 sourcepixels; bilinear downsample30%"}],
        })
    result = {"schema_version": 1, "language": "ja", "evaluation_card_names": ["Opt", "Divination"], "training_membership_note": "Opt appeared in the frozen ocracle LatinV1 corpus; these are pretrained-model test fixtures, not globally unseen canonical identities. Original upstream pretraining membership is unknown.",
              "scope": "Two Japanese printings and a fixed set of horizontal text/furigana lines. No vertical, language-wide or font-wide coverage claim.",
              "samples": samples}
    (output / "fixtures.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(samples)} reviewed Japanese crop/blur pairs")


if __name__ == "__main__":
    main()
