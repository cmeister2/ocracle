#!/usr/bin/env python3
"""Compare enhancement implementations on a fixed, reviewed development cohort."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
MANIFEST = DIRECTORY / "fixtures.json"
RESULTS = DIRECTORY / "results.json"
SOURCE = DIRECTORY / "source.json"
VARIANTS = ("clean", "gaussian", "downsample", "jpeg")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[-1] + 1,
                               previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def prepare():
    from PIL import Image, ImageFilter, __version__
    if __version__ != "13.0.0":
        raise ValueError("Pillow13.0.0 is required for exact fixture reproduction")
    titles_path = ROOT / "training/latin-v2-corpus/titles-reviewed.json"
    bodies_path = ROOT / "training/latin-v2-corpus/body-lines.json"
    titles = [row for row in json.loads(titles_path.read_text())
              if row["review_status"] == "accepted" and row["split"] == "validation"]
    bodies = [row for row in json.loads(bodies_path.read_text())
              if row["review_status"] == "accepted" and row["card_name"] != "Opt"]
    if len(titles) != 23 or len(bodies) != 20:
        raise ValueError("Development cohort changed: expected23 titles and20 body lines")
    (DIRECTORY / "images").mkdir(exist_ok=True)
    records = []
    for index, row in enumerate(titles + bodies):
        scan = ROOT / "training/data/latin-v2/replay-scans" / (row["card_id"] + ".img")
        if not scan.exists():
            scan.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(row["image_url"], headers={"User-Agent": "ocracle-colour-development/1"})
            scan.write_bytes(urllib.request.urlopen(request, timeout=60).read())
        if digest(scan) != row["scan_sha256"]:
            raise ValueError("Source hash mismatch: " + row["card_id"])
        source = Image.open(scan).convert("RGB").resize((672, 936)).crop(row["bbox"])
        samples = {"clean": source, "gaussian": source.filter(ImageFilter.GaussianBlur(2.4))}
        samples["downsample"] = samples["gaussian"].resize(
            (max(1, round(source.width * .3)), max(1, round(source.height * .3))), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        samples["downsample"].save(buffer, format="JPEG", quality=45, subsampling=2)
        samples["jpeg"] = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        case = f"{index:02d}-{row['card_id']}"
        variants = []
        for name, image in samples.items():
            path = DIRECTORY / "images" / f"{case}-{name}.png"
            image.save(path)
            variants.append({"id": name, "path": str(path.relative_to(DIRECTORY)),
                             "sha256": digest(path), "size": list(image.size)})
        records.append({"id": case, "card_id": row["card_id"], "oracle_id": row["oracle_id"],
                        "card_name": row["card_name"], "language": row["language"],
                        "text": row["text"], "kind": row["kind"], "italic": row["italic"],
                        "corpus_split": row["split"], "source_url": row["image_url"],
                        "source_sha256": row["scan_sha256"], "bbox_after_resize_672_936": row["bbox"],
                        "review_source": str((titles_path if index < len(titles) else bodies_path).relative_to(ROOT)),
                        "variants": variants})
    manifest = {"schema_version": 1, "purpose": "development; not an independent test",
                "selection": "Every accepted LatinV2 validation title and all20 reviewed non-Opt body lines. No OCR filtering for this experiment; body discovery originally used accent-folded matching.",
                "review_sources": {str(p.relative_to(ROOT)): digest(p) for p in [titles_path, bodies_path]},
                "preparation": "Pillow13.0.0: RGB source resized672x936 with default bicubic; reviewed bbox; clean, Gaussian sigma2.4, then30% BILINEAR, thenJPEG quality45 subsampling2.",
                "lineage": "Known LatinV2 development/training identities. Upstream Paddle pretraining membership unknown. No frozen gallery/accent/rare/poster tests select this change.",
                "samples": records}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("Reproduction differs from frozen development manifest")
    MANIFEST.write_text(encoded)
    print(f"Prepared {len(records)} reviewed RGB lines and{len(records)*4} variants")


def metrics(records, field):
    result = defaultdict(lambda: {"observations": 0, "exact": 0, "edits": 0, "characters": 0})
    for row in records:
        reference = normalized(row["reference"])
        inference = row[field]
        for mode, prediction in [("selected", inference["text"]),
                                 ("original", inference["candidates"][0]["text"]),
                                 ("enhanced", inference["candidates"][1]["text"])]:
            prediction = normalized(prediction)
            for group in ["all", row["variant"], "italic" if row["italic"] else "regular", row["language"]]:
                counts = result[f"{mode}/{group}"]
                counts["observations"] += 1
                counts["exact"] += int(reference == prediction)
                counts["edits"] += distance(reference, prediction)
                counts["characters"] += len(reference)
    return {key: dict(value, cer=value["edits"] / max(1, value["characters"]))
            for key, value in sorted(result.items())}


def check(results_path=RESULTS):
    manifest = json.loads(MANIFEST.read_text())
    for path, expected in manifest["review_sources"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Reviewed cohort source changed: " + path)
    cases = {}
    for sample in manifest["samples"]:
        if {v["id"] for v in sample["variants"]} != set(VARIANTS):
            raise ValueError("Missing development degradation")
        for variant in sample["variants"]:
            key = sample["id"], variant["id"]
            if key in cases or digest(DIRECTORY / variant["path"]) != variant["sha256"]:
                raise ValueError("Duplicate case or changed image")
            cases[key] = sample, variant
    report = json.loads(results_path.read_text())
    source = json.loads(SOURCE.read_text())
    if any(report.get(key) != value for key, value in source.items()):
        raise ValueError("Report does not match the frozen implementation provenance")
    if report["fixtures_sha256"] != digest(MANIFEST):
        raise ValueError("Report references a different cohort")
    if report["patch_sha256"] != digest(DIRECTORY / "colour-preserving.patch"):
        raise ValueError("Archived enhancement patch changed")
    historical = source["historical_profiles_manifest"]
    if digest(ROOT / historical["path"]) != historical["sha256"]:
        raise ValueError("Historical model/profile manifest changed")
    model_manifest = json.loads((ROOT / historical["path"]).read_text())
    profiles = {entry["id"]: entry for entry in model_manifest["profiles"]}
    if report["model_sha256"] != profiles["en"]["sha256"]:
        raise ValueError("Report used a different recognition model")
    for language in {sample["language"] for sample in manifest["samples"]}:
        profile = profiles[language]
        expected = digest(ROOT / profile["configuration"])
        if (profile["sha256"] != report["model_sha256"]
                or profile["configuration_sha256"] != expected
                or report["configurations"][language + ".json"] != expected):
            raise ValueError("Report configuration differs from its registered profile")
    seen = set()
    for row in report["records"]:
        key = row["id"], row["variant"]
        if key in seen or key not in cases:
            raise ValueError("Unexpected or duplicate observation")
        seen.add(key)
        sample, variant = cases[key]
        if (row["reference"] != sample["text"] or row["image_sha256"] != variant["sha256"]
                or row["language"] != sample["language"] or row["italic"] != sample["italic"]):
            raise ValueError("Observation differs from source fixture")
        for field in ["before", "after"]:
            inference = row[field]
            if inference["model"]["sha256"] != report["model_sha256"] or inference["model"]["language"] != sample["language"]:
                raise ValueError("Wrong model or language")
            if not math.isfinite(inference["confidence"]) or not 0 <= inference["confidence"] <= 1:
                raise ValueError("Invalid confidence")
            candidates = inference["candidates"]
            if len(candidates) != 2:
                raise ValueError("Expected original and enhanced candidates")
            for index, candidate in enumerate(candidates):
                confidence = candidate["confidence"]
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Invalid candidate confidence")
                prefix = "original + " if index == 0 else "2x Lanczos + "
                if not candidate["method"].startswith(prefix):
                    raise ValueError("Original/enhanced candidate order changed")
            # Rust Iterator::max_by returns the last candidate on an exact tie.
            best = max(enumerate(candidates), key=lambda pair: (pair[1]["confidence"], pair[0]))[1]
            if any(inference[field] != best[field] for field in ["text", "confidence", "method", "lines"]):
                raise ValueError("Selected output is not the highest-confidence candidate")
        if row["before"]["candidates"][0] != row["after"]["candidates"][0]:
            raise ValueError("Original-pass output unexpectedly changed")
    if seen != set(cases) or len(cases) != 172:
        raise ValueError("Incomplete development observations")
    for field in ["before", "after"]:
        if report["metrics"][field] != metrics(report["records"], field):
            raise ValueError("Incorrect recorded metrics")
    replay = json.loads((DIRECTORY / "reproduction.json").read_text())
    if (replay["original_report_sha256"] != digest(RESULTS)
            or replay["paired_inputs"] != 172 or replay["actual_cli_calls"] != 344
            or replay["all_raw_inference_records_identical"] is not True
            or replay["all_metrics_identical"] is not True
            or replay["complete_replay_path_including_report_creation_and_offline_verification_passed"] is not True):
        raise ValueError("Frozen full-replay evidence does not match the original report")
    print("Verified172 paired colour-development observations and their original/enhanced candidates")


def run(before, after, output):
    manifest = json.loads(MANIFEST.read_text())
    source = json.loads(SOURCE.read_text())
    if digest(ROOT / "src/engine.rs") != source["source_engine_sha256"]:
        raise ValueError("Apply the archived colour-preserving patch in the documented source checkout first")
    with tempfile.TemporaryDirectory(prefix="ocracle-colour-") as temporary:
        snapshot = Path(temporary)
        executables = {}
        for name, cli_source in [("before", before), ("after", after)]:
            target = snapshot / name
            shutil.copy2(cli_source, target)
            executables[name] = target
        shutil.copytree(ROOT / "models/paddle-v6", snapshot / "models/paddle-v6")
        shutil.copy2(ROOT / "models/profiles.json", snapshot / "models/profiles.json")
        shutil.copy2(ROOT / "models/ppocr-v6-small.onnx", snapshot / "models/ppocr-v6-small.onnx")

        def observe(item):
            sample, variant = item
            path = DIRECTORY / variant["path"]
            if digest(path) != variant["sha256"]:
                raise ValueError("Input changed during experiment")
            row = {"id": sample["id"], "variant": variant["id"], "language": sample["language"],
                   "italic": sample["italic"], "reference": sample["text"], "image_sha256": variant["sha256"]}
            for name, executable in executables.items():
                process = subprocess.run([str(executable), "recognize", str(path), "--line", "--passes", "all",
                                          "--language", sample["language"], "--model-root", str(snapshot)],
                                         text=True, capture_output=True, check=True)
                row[name] = json.loads(process.stdout)
            return row

        inputs = [(sample, variant) for sample in manifest["samples"] for variant in sample["variants"]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = []
            for index, record in enumerate(pool.map(observe, inputs), 1):
                records.append(record)
                if index % 32 == 0:
                    print(f"Compared {index}/{len(inputs)} paired inputs", flush=True)
        report = {"schema_version": 1, "fixtures_sha256": digest(MANIFEST),
                  "model_sha256": digest(snapshot / "models/ppocr-v6-small.onnx"),
                  "executables": {name: digest(path) for name, path in executables.items()},
                  "configurations": {p.name: digest(p) for p in (snapshot / "models/paddle-v6").glob("*.json")},
                  **source,
                  "metrics": {name: metrics(records, name) for name in executables}, "records": records}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    check(output)
    for name, groups in report["metrics"].items():
        print(name, json.dumps({key: value for key, value in groups.items() if key.startswith("selected/")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, default=ROOT / "target/release/ocracle")
    parser.add_argument("--output", type=Path, help="Fresh run output; defaults to /tmp/ocracle-colour-results.json")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        check(args.output or RESULTS)
    elif args.before:
        run(args.before.resolve(), args.after.resolve(), args.output or Path("/tmp/ocracle-colour-results.json"))
    else:
        parser.error("Choose --prepare, --check or --before EXECUTABLE")


if __name__ == "__main__":
    main()
