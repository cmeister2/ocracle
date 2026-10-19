#!/usr/bin/env python3
"""Score a fixed neural checkpoint on the independent printed-accent cases."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def digest(path):
    """Identify the exact artifact used by inference."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    """Normalize whitespace without discarding language orthography."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(left, right):
    """Count character edits while retaining punctuation, case and accents."""
    previous = list(range(len(right) + 1))
    for row, expected in enumerate(left, start=1):
        current = [row]
        for column, observed in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != observed)))
        previous = current
    return previous[-1]


def main():
    """Snapshot files before inference so another training run cannot alter them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognition-model", type=Path, required=True)
    parser.add_argument("--model-config-dir", type=Path, default=ROOT / "models/latin-profiles")
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/ocracle")
    parser.add_argument("--passes", choices=["original", "enhanced", "all"], default="all")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = DIRECTORY / "manifest.jsonl"
    samples = [json.loads(line) for line in manifest.read_text().splitlines()]
    records = []
    groups = defaultdict(lambda: {"observations": 0, "character_edits": 0,
                                  "reference_characters": 0, "exact_matches": 0})
    with tempfile.TemporaryDirectory(prefix="ocracle-accent-score-") as temporary:
        snapshot = Path(temporary)
        model = snapshot / "recognition.rten"
        executable = snapshot / "ocracle"
        shutil.copy2(args.recognition_model, model)
        shutil.copy2(args.executable, executable)
        model_sha = digest(model)
        executable_sha = digest(executable)
        if args.expected_sha256 and model_sha != args.expected_sha256:
            raise ValueError("Checkpoint differs from the requested frozen artifact")
        for language in {sample["language"] for sample in samples}:
            shutil.copy2(args.model_config_dir / (language + ".json"), snapshot / (language + ".json"))
        for sample in samples:
            image = DIRECTORY / sample["path"]
            if digest(image) != sample["image_sha256"]:
                raise ValueError("Frozen evaluation image changed: " + sample["path"])
            config = snapshot / (sample["language"] + ".json")
            alphabet = set(json.loads(config.read_text())["alphabet"])
            process = subprocess.run([
                str(executable), "recognize", str(image), "--recognition-model", str(model),
                "--model-config", str(config), "--line", "--passes", args.passes,
            ], capture_output=True, text=True, check=True)
            inference = json.loads(process.stdout)
            if inference["model"]["sha256"] != model_sha:
                raise ValueError("Inference reported a different checkpoint")
            if not math.isfinite(inference["confidence"]) or not 0 <= inference["confidence"] <= 1:
                raise ValueError("Invalid inference confidence")
            expected = normalized(sample["text"])
            observed = normalized(inference["text"])
            edits = distance(expected, observed)
            records.append({
                "card_id": sample["card_id"], "language": sample["language"],
                "severity": sample["severity"], "image": sample["path"],
                "image_sha256": sample["image_sha256"], "reference": sample["text"],
                "normalized_prediction": observed, "character_edits": edits,
                "reference_characters": len(expected), "exact_match": expected == observed,
                "missing_checkpoint_alphabet": sorted(set(expected) - alphabet),
                "config_sha256": digest(config), "inference": inference,
            })
            for key in ["overall", "language:" + sample["language"], "severity:" + sample["severity"]]:
                group = groups[key]
                group["observations"] += 1
                group["character_edits"] += edits
                group["reference_characters"] += len(expected)
                group["exact_matches"] += int(expected == observed)
            print(f"{sample['language']} {sample['severity']}: {sample['text']!r} -> {observed!r}; edits {edits}", flush=True)
    for group in groups.values():
        group["character_error_rate"] = group["character_edits"] / group["reference_characters"]
    result = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "recognition_model_sha256": model_sha, "executable_sha256": executable_sha,
        "manifest_sha256": digest(manifest), "passes": args.passes,
        "normalization": "NFC and collapsed whitespace, preserving case/punctuation/accents. Raw inference retained.",
        "confidence_note": "Uncalibrated emitted-character posterior score, not probability of an exact transcription.",
        "checkpoint_selection": "This script evaluates a supplied fixed artifact; it never trains or selects weights/configurations.",
        "groups": dict(groups), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["groups"]["overall"], indent=2))


if __name__ == "__main__":
    main()
