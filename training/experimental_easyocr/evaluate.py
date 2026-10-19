#!/usr/bin/env python3
"""Measure actual Rust/RTen OCR on reviewed Japanese print crops and severe blur."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    """Hash an actual model, configuration, executable or fixture artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    """Preserve fullwidth digits, case and punctuation while normalizing whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def distance(reference, observed):
    """Count Unicode-character Levenshtein edits in linear memory."""
    previous = list(range(len(observed) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(observed, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def main():
    """Record both pretrained Japanese candidates without selecting by references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=ROOT / "target/release/ocracle")
    parser.add_argument("--fixtures", type=Path, default=HERE / "fixtures/fixtures.json")
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--models", nargs="+", default=["japanese_g1", "japanese_g2"])
    parser.add_argument("--output", type=Path, default=HERE / "japanese-report.json")
    args = parser.parse_args()
    executable_hash = sha256(args.executable)
    fixtures_hash = sha256(args.fixtures)
    fixtures = json.loads(args.fixtures.read_text())
    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = "4"
    records = []
    model_metadata = {}
    aggregate = defaultdict(lambda: {"observations": 0, "characters": 0, "character_errors": 0, "exact_matches": 0})
    for model in args.models:
        model_path = args.artifacts / f"{model}.onnx"
        config_path = model_path.with_suffix(".json")
        config = json.loads(config_path.read_text())
        model_hash = sha256(model_path)
        config_hash = sha256(config_path)
        model_metadata[model] = {"model_sha256": model_hash, "configuration_sha256": config_hash,
                                 "provenance": json.loads(model_path.with_suffix(".provenance.json").read_text())}
        for sample in fixtures["samples"]:
            expected = normalized(sample["ground_truth"])
            missing = sorted(set(expected) - set(config["alphabet"]), key=ord)
            for variant in sample["variants"]:
                image = args.fixtures.parent / variant["image"]
                command = [str(args.executable.resolve()), "recognize", str(image.resolve()),
                           "--recognition-model", str(model_path.resolve()), "--model-config", str(config_path.resolve()),
                           "--line", "--passes", "all"]
                process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180, env=environment)
                inference = json.loads(process.stdout)
                if inference["model"]["sha256"] != model_hash or inference["model"]["language"] != sample["language"]:
                    raise ValueError("Rust reported a different checkpoint or language profile")
                observed = normalized(inference["text"])
                edits = distance(expected, observed)
                record = {"model": model, "sample_id": sample["id"], "variant_id": variant["id"],
                          "image_sha256": sha256(image), "reference": sample["ground_truth"],
                          "reference_characters": len(expected), "character_errors": edits,
                          "character_error_rate": edits / max(len(expected), 1), "exact_match": expected == observed,
                          "reference_alphabet_gaps": missing, "inference": inference}
                records.append(record)
                for group in [f"{model}/all", f"{model}/{variant['id']}", f"{model}/treatment:{sample['treatment']}"]:
                    item = aggregate[group]
                    item["observations"] += 1
                    item["characters"] += len(expected)
                    item["character_errors"] += edits
                    item["exact_matches"] += expected == observed
                print(f"{model}/{sample['id']}/{variant['id']}: {observed!r} | CER {record['character_error_rate']:.1%} | score {inference['confidence']:.4f}", flush=True)
        if model_hash != sha256(model_path) or config_hash != sha256(config_path):
            raise ValueError("A model input changed during evaluation")
    for item in aggregate.values():
        item["character_error_rate"] = item["character_errors"] / max(item["characters"], 1)
    if fixtures_hash != sha256(args.fixtures) or executable_hash != sha256(args.executable):
        raise ValueError("An inference input changed during evaluation")
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "executable_sha256": executable_hash,
              "fixtures_sha256": fixtures_hash, "model_metadata": model_metadata, "groups": dict(aggregate),
              "confidence_note": "Raw, uncalibrated geometric mean of emitted CTC character posterior peaks; not whole-line accuracy. Candidate preprocessing selection uses model score, never the reference text.",
              "metric_note": "NFC Unicode normalization and collapsed whitespace; fullwidth digits and punctuation count. Reported scores reflect actual horizontal Japanese crops only.",
              "coverage_limitations": ["Only two card identities; not a language-wide accuracy estimate", "No MTG fine-tuning", "Both upstream Japanese alphabets miss printed corpus characters", "Japanese flavor/reminder text is upright here, not evidence of Latin italic coverage", "Furigana is isolated manually; mixed furigana/main-line segmentation is unverified", "Vertical Japanese remains unverified"],
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded {len(records)} actual Rust/RTen observations: {args.output}", flush=True)


if __name__ == "__main__":
    main()
