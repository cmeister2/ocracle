#!/usr/bin/env python3
"""Audit localized MTG character coverage and export actual EasyOCR CRNN weights.

Uses complete pretrained models, not randomly initialized whole-script heads.
Exports remain experimental until actual card evaluation and RTen parity pass.
"""

import argparse
import json
from pathlib import Path

from experimental_easyocr.pipeline import HERE, PROFILE_MODELS, audit_corpus, export_model


def main():
    """Audit a corpus and optionally export the preferred or explicitly named model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROFILE_MODELS), default="ja")
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus/cards.json")
    parser.add_argument("--checkpoint", help="Explicit upstream model; default is best corpus alphabet coverage")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-alphabet-gaps", action="store_true", help="Permit evaluation exports with explicit unsupported characters")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_corpus(args.corpus, args.profile)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return
    checkpoint_name = args.checkpoint or audit["preferred_model_by_coverage"]
    candidate = next((candidate for candidate in audit["candidates"] if candidate["model"] == checkpoint_name), None)
    if candidate is None:
        parser.error("Checkpoint is not a candidate for this language profile")
    if candidate["missing_unique_characters"] and not args.allow_alphabet_gaps:
        parser.error("Pretrained alphabet cannot emit every corpus character; --allow-alphabet-gaps permits an explicitly incomplete evaluation export")
    output = args.output or HERE / "artifacts" / f"{checkpoint_name}.onnx"
    provenance = export_model(checkpoint_name, args.cache, output, args.profile, audit)
    print(json.dumps({"output": str(output), "sha256": provenance["onnx_sha256"],
                      "checkpoint": checkpoint_name, "experimental": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
