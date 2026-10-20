#!/usr/bin/env python3
"""Prepare pinned OCR workloads and run the release latency harness sequentially.

Requires Pillow. Run ``--prepare`` first; cached full scans are checked against
the existing gallery source hashes, or fetched with ``--download-sources``.
No model output selects a fixture, reference, or scored region.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import urllib.request

from PIL import Image

from gallery_export import edit_distance, normalized

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/performance/fixtures/manifest.json"
GALLERY = ROOT / "docs/gallery/assets/fixtures.json"
LATIN_GALLERY = ROOT / "docs/gallery/assets/latin/fixtures.json"

# Manually transcribed from the hash-pinned original scans on 2026-10-20.
# Body bounds are also the exact multiline crop. No reference is inferred from
# a recognizer or modern Oracle text. Symbols/art/collector text are unscored.
CARDS = (
    {
        "id": "divination", "sample": "beleren-title", "cache": "m15-52.png",
        "regions": [
            ("title", [40, 40, 420, 115], "Divination"),
            ("type", [40, 580, 580, 645], "Sorcery"),
            ("body", [52, 690, 688, 925],
             "Draw two cards. “The key to unlocking this puzzle is within you.” "
             "—Doriel, mentor of Mistral Isle"),
        ],
    },
    {
        "id": "air-elemental", "sample": "matrix-title", "cache": "8ed-59.png",
        "regions": [
            ("title", [45, 40, 480, 115], "Air Elemental"),
            ("type", [45, 580, 590, 645], "Creature — Elemental"),
            ("body", [65, 680, 675, 900],
             "Flying (This creature can’t be blocked except by creatures with flying.) "
             "Pray that it doesn’t seek the safety of your lungs."),
        ],
    },
    {
        "id": "llanowar-elves", "sample": "goudy-title", "cache": "lea-210.png",
        "regions": [
            ("title", [35, 35, 500, 100], "Llanowar Elves"),
            ("type", [65, 570, 590, 617], "Summon Elves"),
            ("body", [108, 625, 653, 893],
             "Tap to add 1 green mana to your mana pool. This tap can be played as an interrupt. "
             "Whenever the Llanowar Elves gather the fruits of their forest, they leave one plant "
             "of each type untouched, considering that nature’s portion."),
        ],
    },
)


def sha256(path: Path) -> str:
    """Return the exact artifact identity."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    """Use checkout-relative paths when possible, absolute paths otherwise."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file atomically after each completed observation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def image_fields(path: Path) -> dict:
    """Record input bytes and dimensions independently of the inference binary."""
    with Image.open(path) as image:
        size = list(image.size)
    return {"image": relative(path), "image_sha256": sha256(path), "image_size": size}


def prepare(manifest_path: Path, download: bool) -> None:
    """Create full-card, multiline and existing gallery line workloads."""
    samples = []
    for gallery_path in (GALLERY, LATIN_GALLERY):
        gallery = json.loads(gallery_path.read_text())
        for sample in gallery["samples"]:
            for variant in sample["variants"]:
                samples.append({
                    "id": f"line-{sample['id']}-{variant['id']}",
                    "workload": "line", "mode": "line",
                    "language": sample["language"],
                    "source": sample["source"], "variant": variant["id"],
                    "ground_truth": sample["ground_truth"],
                    "scoring": "complete crop text; NFC and collapsed whitespace",
                    **image_fields(gallery_path.parent / variant["image"]),
                })
    english = {sample["id"]: sample for sample in json.loads(GALLERY.read_text())["samples"]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for card in CARDS:
        source = english[card["sample"]]["source"]
        path = ROOT / ".cache/gallery" / card["cache"]
        if not path.exists() and download:
            request = urllib.request.Request(source["image_url"], headers={
                "User-Agent": "ocracle-performance/0.1 (+https://github.com/cmeister2/ocracle)",
                "Accept": "image/png",
            })
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["source_image_sha256"]:
                raise ValueError(f"Source scan changed: {source['image_url']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if not path.exists():
            raise FileNotFoundError(f"{path}: pass --download-sources to fetch the pinned scan")
        if sha256(path) != source["source_image_sha256"]:
            raise ValueError(f"Cached scan differs from frozen gallery source: {path}")
        with Image.open(path) as image:
            if image.size != (745, 1040):
                raise ValueError(f"Unexpected scan dimensions: {path}")
            body = next(region for region in card["regions"] if region[0] == "body")
            crop_path = manifest_path.parent / f"{card['id']}-body.png"
            image.convert("RGB").crop(body[1]).save(crop_path, optimize=True)
        samples.append({
            "id": f"multiline-{card['id']}", "workload": "multiline", "mode": "full",
            "language": "en", "source": source, "variant": "clean",
            "crop_bounds_xyxy": body[1], "ground_truth": body[2],
            "scoring": "complete manually transcribed body crop; NFC and collapsed whitespace",
            **image_fields(crop_path),
        })
        samples.append({
            "id": f"card-{card['id']}", "workload": "card", "mode": "full",
            "language": "en", "source": source, "variant": "clean",
            "scoring": "partial card: fixed title/type/body regions only; "
                       "mana, artwork and collector text unscored; all inference is timed",
            "regions": [{"id": name, "bounds_xyxy": bounds, "text": text}
                        for name, bounds, text in card["regions"]],
            **image_fields(path),
        })
    write_json(manifest_path, {
        "schema_version": 1,
        "notes": "Clean full scans are not camera frames; no motion, glare or thermal coverage. "
                 "Fixed gallery identities excluded from custom fine-tuning; upstream training "
                 "membership unknown. Whole-card CER covers only fixed annotated regions. "
                 "Card imagery/text copyright Wizards of the Coast; scans via Scryfall.",
        "source_documents": {relative(path): sha256(path) for path in (GALLERY, LATIN_GALLERY)},
        "transcription_reviewed_on": "2026-10-20", "samples": samples,
    })
    print(f"Prepared {len(samples)} fixtures: 50 gallery lines, 3 body crops, 3 whole cards")


def text_score(reference: str, observed: str) -> dict:
    """Score literal punctuation, case and diacritics after documented normalization."""
    reference, observed = normalized(reference), normalized(observed)
    edits = edit_distance(reference, observed)
    return {
        "reference": reference, "observed": observed,
        "character_edits": edits, "reference_characters": len(reference),
        "character_error_rate": edits / max(len(reference), 1),
        "exact_match": reference == observed,
    }


def score(sample: dict, inference: dict) -> dict:
    """Evaluate full crop text, or select full-card lines by fixed region centers."""
    if "regions" not in sample:
        return text_score(sample["ground_truth"], inference["text"])
    regions, used = [], set()
    for region in sample["regions"]:
        x0, y0, x1, y1 = region["bounds_xyxy"]
        selected = []
        for index, line in enumerate(inference["lines"]):
            left, top, width, height = line["bounds"]
            if x0 <= left + width / 2 < x1 and y0 <= top + height / 2 < y1:
                if index in used:
                    raise ValueError("Scoring regions overlap at a predicted line center")
                selected.append(line["text"])
                used.add(index)
        regions.append({
            "id": region["id"], "bounds_xyxy": region["bounds_xyxy"],
            **text_score(region["text"], " ".join(selected)),
        })
    edits = sum(region["character_edits"] for region in regions)
    characters = sum(region["reference_characters"] for region in regions)
    return {
        "regions": regions, "character_edits": edits, "reference_characters": characters,
        "character_error_rate": edits / max(characters, 1),
        "exact_match": all(region["exact_match"] for region in regions),
        "unscored_prediction_lines": len(inference["lines"]) - len(used),
        "assignment": "line center inside fixed region; predicted reading order preserved",
    }


def check_inference(sample: dict, inference: dict, iterations: int) -> None:
    """Reject mismatched images, invalid timings and out-of-image coordinates."""
    if inference["image_size"] != sample["image_size"]:
        raise ValueError(f"Image dimensions disagree for {sample['id']}")
    timings = inference["samples_ms"]
    if len(timings) != iterations or any(not math.isfinite(t) or t <= 0 for t in timings):
        raise ValueError(f"Invalid measured samples for {sample['id']}")
    ordered = sorted(timings)
    for field, expected in (
        ("median_ms", ordered[len(ordered) // 2]),
        ("p95_ms", ordered[math.ceil(len(ordered) * 0.95) - 1]),
    ):
        if not math.isclose(inference[field], expected, rel_tol=1e-9):
            raise ValueError(f"{field} differs from raw samples for {sample['id']}")
    width, height = sample["image_size"]
    for line in inference["lines"]:
        left, top, box_width, box_height = line["bounds"]
        if not (0 <= left <= left + box_width <= width and
                0 <= top <= top + box_height <= height):
            raise ValueError(f"Line bounds outside original coordinates for {sample['id']}")


def run(args: argparse.Namespace) -> None:
    """Run one engine configuration over selected fixtures without parallelism."""
    manifest = json.loads(args.manifest.read_text())
    for name, expected in manifest["source_documents"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Fixture source manifest changed: {name}")
    config = json.loads(args.config.read_text())
    if args.language not in config.get("languages", [config["language"]]):
        raise ValueError("Model configuration language differs from selected fixture language")
    selected = [sample for sample in manifest["samples"]
                if sample["language"] == args.language
                and (not args.workload or sample["workload"] in args.workload)
                and (not args.sample or sample["id"] in args.sample)]
    if not selected:
        raise ValueError("No fixtures match the requested filters")
    if args.sample and set(args.sample) != {sample["id"] for sample in selected}:
        raise ValueError("Some requested sample IDs are absent or excluded by other filters")
    environment = dict(os.environ)
    # Profiling produces output and changes runtime behavior; keep benchmark runs clean.
    for name in ("RTEN_TIMING", "RTEN_TRACE"):
        environment.pop(name, None)
    environment.update(RTEN_NUM_THREADS=str(args.threads), OCRACLE_BACKEND=args.backend)
    if args.detector:
        environment["OCRACLE_DETECTOR_MODEL"] = str(args.detector.resolve())
    else:
        environment.pop("OCRACLE_DETECTOR_MODEL", None)
    detector = args.detector or ROOT / "models/text-detection.onnx"
    result = {
        "schema_version": 1, "label": args.label, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "warm public OCR call; model loading and image decoding excluded; "
                             "desktop timings do not establish Android FPS",
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "processor": platform.processor()},
        "manifest": relative(args.manifest), "manifest_sha256": sha256(args.manifest),
        "executable": str(args.executable.resolve()), "executable_sha256": sha256(args.executable),
        "model": relative(args.model), "model_sha256": sha256(args.model),
        "configuration": relative(args.config), "configuration_sha256": sha256(args.config),
        "detector": relative(detector), "detector_sha256": sha256(detector),
        "threads": args.threads, "layout": args.layout, "backend": args.backend, "iterations": args.iterations,
        "requested_samples": [sample["id"] for sample in selected], "records": [],
    }
    if args.backend == "onnxruntime":
        native = environment.get("ORT_DYLIB_PATH")
        if not native:
            raise ValueError("Set ORT_DYLIB_PATH for reproducible native-library identity")
        native_path = Path(native).resolve()
        environment["ORT_DYLIB_PATH"] = str(native_path)
        result["runtime_library"] = {"path": str(native_path), "sha256": sha256(native_path)}
    write_json(args.output, result)
    for sample in selected:
        path = ROOT / sample["image"]
        if sha256(path) != sample["image_sha256"]:
            raise ValueError(f"Fixture bytes changed: {path}")
        # A standard-card artwork mask is destructive on a pre-cropped text body.
        effective_layout = args.layout if sample["workload"] == "card" else "text"
        environment["OCRACLE_LAYOUT"] = effective_layout
        command = [str(args.executable.resolve()), str(args.model.resolve()),
                   str(args.config.resolve()), str(path.resolve()),
                   str(args.iterations), sample["mode"]]
        print(f"{args.label}: {sample['id']}", flush=True)
        completed = subprocess.run(command, env=environment, cwd=ROOT, check=True,
                                   capture_output=True, text=True)
        inference = json.loads(completed.stdout)
        check_inference(sample, inference, args.iterations)
        if inference.get("backend", "rten") != args.backend:
            raise ValueError("Harness ignored requested inference backend")
        if inference.get("layout") != effective_layout:
            raise ValueError("Harness ignored requested layout or does not report its layout")
        if inference["model"]["sha256"] != result["model_sha256"]:
            raise ValueError("Harness recognition model differs from requested artifact")
        if sample["mode"] == "full":
            if inference["model"]["detector_sha256"] != result["detector_sha256"]:
                raise ValueError("Harness ignored requested detector")
        result["records"].append({
            "sample": sample, "inference": inference, "evaluation": score(sample, inference),
            "command": command, "effective_layout": effective_layout,
        })
        write_json(args.output, result)
    result["status"] = "complete"
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)


def main() -> None:
    """Expose fixture preparation separately from expensive sequential benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--download-sources", action="store_true")
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/examples/latency")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--layout", choices=["text", "standard_card"], default="text",
                        help="Layout for whole cards; line/multiline crops always use text")
    parser.add_argument("--backend", choices=["rten", "onnxruntime"], default="rten")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--language", default="en")
    parser.add_argument("--workload", choices=["line", "multiline", "card"], action="append")
    parser.add_argument("--sample", action="append")
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.manifest, args.download_sources)
        return
    if not all((args.model, args.config, args.label, args.output)):
        parser.error("benchmark runs require --model, --config, --label and --output")
    if args.threads < 1 or args.iterations < 1:
        parser.error("--threads and --iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
