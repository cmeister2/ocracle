#!/usr/bin/env python3
"""Compare all pinned candidates on every frozen development input, without fitting."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import unicodedata

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[1]
FIXTURES = REPO / "training/experimental_colour/fixtures.json"
RESULTS = DIRECTORY / "results-colour-development.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def metrics(records):
    groups = defaultdict(lambda: {"observations": 0, "exact": 0, "edits": 0, "characters": 0})
    for row in records:
        for group in ["all", row["variant"], row["language"], row["kind"], "italic" if row["italic"] else "regular"]:
            counts = groups[row["model"] + "/" + group]
            counts["observations"] += 1
            counts["characters"] += len(normalized(row["reference"]))
            counts["edits"] += distance(normalized(row["reference"]), normalized(row["inference"]["text"]))
            counts["exact"] += normalized(row["reference"]) == normalized(row["inference"]["text"])
    return {key: {**value, "cer": value["edits"] / value["characters"]} for key, value in sorted(groups.items())}


def candidates():
    source = json.loads((DIRECTORY / "sources.json").read_text())
    base = json.loads((DIRECTORY / "configs/v6-small.json").read_text())
    rows = [{"id": "v6-small", "model": REPO / "models/ppocr-v6-small.onnx", "config": base,
             "source_revision": "b8f84f0b80c529de40b4fbb3544b84fa7233a513"}]
    for candidate in source["candidates"]:
        config = json.loads((DIRECTORY / "configs" / (candidate["id"] + ".json")).read_text())
        rows.append({"id": candidate["id"], "model": DIRECTORY / candidate["files"]["inference.onnx"]["path"],
                     "config": config, "source_revision": candidate["revision"]})
    return rows


def check():
    report = json.loads(RESULTS.read_text())
    source = json.loads((DIRECTORY / "sources.json").read_text())
    pins = {source["baseline"]["id"]: source["baseline"]["model_sha256"],
            **{x["id"]: x["files"]["inference.onnx"]["sha256"] for x in source["candidates"]}}
    if set(report["models"]) != set(pins):
        raise ValueError("Candidate set differs from predeclared source pins")
    templates = {name: json.loads((DIRECTORY / "configs" / (name + ".json")).read_text()) for name in pins}
    for name, template in templates.items():
        encoded = json.dumps(template, ensure_ascii=False, indent=2) + "\n"
        if (report["models"][name]["model_sha256"] != pins[name]
                or report["models"][name]["configuration_template_sha256"] != hashlib.sha256(encoded.encode()).hexdigest()):
            raise ValueError("Candidate model or tensor configuration changed")
    if digest(FIXTURES) != report["fixture_manifest_sha256"]:
        raise ValueError("Development fixture manifest changed")
    manifest = json.loads(FIXTURES.read_text())
    cases = {(sample["id"], variant["id"]): (sample, variant)
             for sample in manifest["samples"] for variant in sample["variants"]}
    expected = {(model, *case) for model in report["models"] for case in cases}
    seen = set()
    for row in report["records"]:
        key = row["model"], row["id"], row["variant"]
        if key in seen or key not in expected:
            raise ValueError("Unexpected or duplicate candidate observation")
        seen.add(key)
        sample, variant = cases[row["id"], row["variant"]]
        if digest(FIXTURES.parent / variant["path"]) != variant["sha256"] or row["image_sha256"] != variant["sha256"]:
            raise ValueError("Changed development image")
        if any(row[field] != sample[field] for field in ["language", "kind", "italic", "card_id", "oracle_id"]):
            raise ValueError("Observation identity differs from fixture")
        if row["reference"] != sample["text"] or json.loads(row["stdout"]) != row["inference"]:
            raise ValueError("Reference or raw inference mismatch")
        serialized = json.dumps({**templates[row["model"]], "language": row["language"]}, ensure_ascii=False, indent=2) + "\n"
        if row["configuration_sha256"] != hashlib.sha256(serialized.encode()).hexdigest():
            raise ValueError("Observation configuration differs from pinned template")
        inference = row["inference"]
        if inference["model"]["sha256"] != report["models"][row["model"]]["model_sha256"] or inference["model"]["language"] != row["language"]:
            raise ValueError("Wrong candidate model or language")
        if len(inference["candidates"]) != 1 or not inference["method"].startswith("original"):
            raise ValueError("Original-only evaluation contract changed")
    if seen != expected or report["summary"] != metrics(report["records"]):
        raise ValueError("Missing cases or incorrect metrics")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=REPO / "target/release/ocracle")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    manifest_sha = digest(FIXTURES)
    manifest = json.loads(FIXTURES.read_text())
    cache = DIRECTORY / "cache"
    cache.mkdir(exist_ok=True)
    executable = cache / "ocracle-candidate-snapshot"
    shutil.copy2(args.cli, executable)
    configs = cache / "configs"
    configs.mkdir(exist_ok=True)
    models = {}
    tasks = []
    for model in candidates():
        source = model["model"]
        model_sha = digest(source)
        pins = json.loads((DIRECTORY / "sources.json").read_text())
        expected_sha = (pins["baseline"]["model_sha256"] if model["id"] == "v6-small" else
                        next(x for x in pins["candidates"] if x["id"] == model["id"])["files"]["inference.onnx"]["sha256"])
        if model_sha != expected_sha:
            raise ValueError("Model bytes differ from pinned official source")
        snapshot = cache / (model["id"] + "-snapshot.onnx")
        shutil.copy2(source, snapshot)
        config_json = json.dumps(model["config"], ensure_ascii=False, indent=2) + "\n"
        models[model["id"]] = {"model_sha256": model_sha, "source_revision": model["source_revision"],
                              "model_bytes": source.stat().st_size,
                              "configuration_template_sha256": hashlib.sha256(config_json.encode()).hexdigest()}
        for language in sorted({sample["language"] for sample in manifest["samples"]}):
            (configs / (model["id"] + "-" + language + ".json")).write_text(
                json.dumps({**model["config"], "language": language}, ensure_ascii=False, indent=2) + "\n")
        for sample in manifest["samples"]:
            for variant in sample["variants"]:
                image = FIXTURES.parent / variant["path"]
                if digest(image) != variant["sha256"]:
                    raise ValueError("Development image changed")
                tasks.append((model["id"], snapshot, sample, variant, image))

    def infer(task):
        model_id, model, sample, variant, image = task
        config = configs / (model_id + "-" + sample["language"] + ".json")
        command = [str(executable), "recognize", str(image), "--recognition-model", str(model),
                   "--model-config", str(config), "--line", "--passes", "original"]
        started = time.monotonic()
        process = subprocess.run(command, check=True, capture_output=True, text=True,
                                 env={**os.environ, "RAYON_NUM_THREADS": "2"})
        row = {"model": model_id, "id": sample["id"], "variant": variant["id"],
               **{key: sample[key] for key in ["card_id", "oracle_id", "language", "kind", "italic"]},
               "reference": sample["text"], "image_sha256": variant["sha256"],
               "configuration_sha256": digest(config), "wall_seconds": time.monotonic() - started,
               "stdout": process.stdout, "stderr": process.stderr, "inference": json.loads(process.stdout)}
        print(model_id, sample["id"], variant["id"], row["inference"]["text"], flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(infer, tasks))
    if manifest_sha != digest(FIXTURES):
        raise ValueError("Manifest changed during inference")
    report = {"schema_version": 1, "evaluated_on": "2026-10-19", "status": "development_only_no_fitting",
              "fixture_manifest": str(FIXTURES.relative_to(REPO)), "fixture_manifest_sha256": manifest_sha,
              "cli_sha256": digest(executable), "rten_version": "0.26.0", "models": models,
              "selection": "All43 previously reviewed development lines and all4 fixed degradations; all3 models chosen before inference. No frozen gallery, accent, rare or poster test cases.",
              "comparison": "Actual snapshotted Rust CLI; original pass only; BGR image-crate Triangle resize; NFC plus collapsed whitespace, retaining case, accents and punctuation.",
              "limitations": "Known LatinV2 development/training identities, originally CRNN-biased body discovery; unknown upstream Paddle training membership. Model sizes and process timings are not throughput benchmarks.",
              "summary": metrics(records), "records": records}
    RESULTS.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    check()


if __name__ == "__main__":
    main()
