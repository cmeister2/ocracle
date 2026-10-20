#!/usr/bin/env python3
"""Fetch hash-pinned mobile ONNX candidates and reproduce experiment configs.

Requires PyYAML (pinned in training/experimental_paddle/requirements.txt).
Downloads only original publisher artifacts and performs no model inference.
Registry/default models are unchanged; outputs go to a separate cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/performance/mobile-models.json"
DEFAULT_OUTPUT = ROOT / ".cache/performance/mobile-models"


def sha256(data: bytes) -> str:
    """Identify the complete bytes of an original or generated artifact."""
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    """Use a stable encoding shared with the recorded experiment configs."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def check_bytes(data: bytes, expected: dict, label: str) -> None:
    """Reject corrupted downloads or a cache entry from another revision."""
    if len(data) != expected["bytes"] or sha256(data) != expected["sha256"]:
        raise ValueError(f"Size or SHA-256 mismatch: {label}; existing files are not overwritten")


def write_or_check(path: Path, data: bytes) -> None:
    """Create a new artifact or require an existing one to be identical."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing generated artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def fetch_file(directory: Path, filename: str, expected: dict, offline: bool) -> bytes:
    """Verify cached bytes, or verify an immutable download before saving it."""
    path = directory / filename
    if path.exists():
        data = path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Required cached artifact is absent: {path}")
        request = urllib.request.Request(
            expected["url"], headers={"User-Agent": "ocracle-mobile-model-reproduction/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            # Reject a surprising response before retaining an unbounded download.
            data = response.read(expected["bytes"] + 1)
        check_bytes(data, expected, expected["url"])
        write_or_check(path, data)
    check_bytes(data, expected, str(path))
    return data


def recognizer_configuration(model: dict, settings: dict) -> dict:
    """Retain every dictionary class and reproduce the existing pixel contract."""
    transforms = settings["PreProcess"]["transform_ops"]
    decode = next(op["DecodeImage"] for op in transforms if "DecodeImage" in op)
    resize = next(op["RecResizeImg"] for op in transforms if "RecResizeImg" in op)
    if decode != {"channel_first": False, "img_mode": "BGR"}:
        raise ValueError("Unexpected recognizer input channel contract")
    if resize["image_shape"] != [3, 48, 320]:
        raise ValueError("Unexpected recognizer input dimensions")
    postprocess = settings["PostProcess"]
    if postprocess["name"] != "CTCLabelDecode":
        raise ValueError("Unexpected recognizer decoder")
    dictionary = postprocess["character_dict"]
    if not isinstance(dictionary, list) or not all(
        isinstance(character, str) and len(character) == 1 for character in dictionary
    ):
        raise ValueError("Dictionary must contain one Unicode scalar per class")
    graph = model["graph"]
    if len(dictionary) + 2 != graph["ctc_classes"]:
        raise ValueError("Dictionary, appended space and blank do not match graph classes")
    if len(set(dictionary)) != graph["unique_dictionary_entries"]:
        raise ValueError("Dictionary multiplicity differs from the inspected graph contract")
    identity = model["configuration"]
    return {
        "model_name": identity["model_name"],
        "architecture": identity["architecture"],
        "language": identity["language"],
        # Duplicate glyphs belong to different trained classes; keep all indices.
        "alphabet": "".join(dictionary) + " ",
        "input_height": 48,
        "input": {
            "color_mode": "bgr",
            "scale": 2 / 255,
            "offset": -1,
            "padding_width": 320,
            "min_content_width": 1,
            "width_rounding": "ceil",
        },
        "output_layout": "NTC",
        "output_values": "probabilities",
        "dictionary_source": (
            f"Official inference.yml SHA-256 {model['files']['inference.yml']['sha256']}, "
            "original dictionary order then appended space; blank index 0"
        ),
    }


def validate_manifest(manifest: dict) -> None:
    """Require known local output names and immutable, consistently pinned URLs."""
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported mobile model manifest version")
    identifiers = set()
    for model in manifest["models"]:
        name = model["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in identifiers:
            raise ValueError(f"Invalid or duplicate model identifier: {name}")
        identifiers.add(name)
        if model["repository"] != f"PaddlePaddle/{name}":
            raise ValueError("Unexpected model publisher/repository")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("Model revision must be a complete immutable commit")
        if set(model["files"]) != {"README.md", "inference.yml", "inference.onnx"}:
            raise ValueError("Unexpected candidate artifact list")
        for filename, expected in model["files"].items():
            url = (
                f"https://huggingface.co/{model['repository']}/resolve/"
                f"{model['revision']}/{filename}"
            )
            if expected["url"] != url or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]):
                raise ValueError("Artifact URL or SHA-256 is not pinned")
            if not isinstance(expected["bytes"], int) or expected["bytes"] <= 0:
                raise ValueError("Invalid artifact size")
            published = expected.get("published_lfs_sha256")
            if published is not None and published != expected["sha256"]:
                raise ValueError("Publisher LFS and recorded SHA-256 disagree")


def main() -> None:
    """Fetch selected candidates without modifying weights or registered models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model", action="append", default=[],
        help="Manifest model ID; repeat to select several (default: all five)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Require cached publisher files; still reproduce/verify generated configs",
    )
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    known = {model["id"] for model in manifest["models"]}
    if set(args.model) - known:
        parser.error(f"Unknown model IDs: {', '.join(sorted(set(args.model) - known))}")
    try:
        import yaml
    except ImportError as error:
        raise SystemExit(
            "PyYAML is required; install PyYAML==6.0.3 from the experimental Paddle requirements."
        ) from error

    selected = set(args.model) or known
    for model in manifest["models"]:
        if model["id"] not in selected:
            continue
        directory = args.output_dir / model["id"]
        directory.mkdir(parents=True, exist_ok=True)
        originals = {
            filename: fetch_file(directory, filename, expected, args.offline)
            for filename, expected in model["files"].items()
        }
        settings = yaml.safe_load(originals["inference.yml"])
        generated = {}
        if model["kind"] == "recognizer":
            config = json_bytes(recognizer_configuration(model, settings))
            expected = model["configuration"]
            if sha256(config) != expected["sha256"]:
                raise ValueError("Generated recognizer configuration differs from its recorded SHA-256")
            write_or_check(directory / expected["filename"], config)
            generated[expected["filename"]] = {"bytes": len(config), "sha256": sha256(config)}
        else:
            # The original DB parameters are preserved in the manifest/YAML.
            # A detector adapter must choose and record its resize policy.
            if settings["PreProcess"] != model["published_preprocessing"]:
                raise ValueError("Unexpected detector preprocessing")
            if settings["PostProcess"] != model["published_postprocessing"]:
                raise ValueError("Unexpected detector postprocessing")
        report = {
            "schema_version": 1,
            "manifest_sha256": sha256(manifest_bytes),
            "candidate": model,
            "generated_files": generated,
            "source_graph_modified": False,
            "inference_performed": False,
        }
        write_or_check(directory / "ocracle-candidate.json", json_bytes(report))
        print(json.dumps({
            "model": model["id"], "directory": str(directory),
            "model_sha256": model["files"]["inference.onnx"]["sha256"],
            "generated_files": generated,
        }))


if __name__ == "__main__":
    main()
