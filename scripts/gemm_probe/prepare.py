#!/usr/bin/env python3
"""Prepare an isolated RTen 0.26 GEMM probe without modifying project dependencies.

Copies the cached crate and changes only the visibility of its internal kernel
selector. Building and running are separate, so benchmarks can be scheduled
without CPU contention. The generic implementation may be auto-vectorized and
must not be labeled scalar.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


REVISION = "14cbf4620476f402237bb29f731a244d3b6bed4d"
ARCHIVE_SHA256 = "8efc07c654d4b3f8d81e2efd8b13aea7475568395289e5bb35393411aa6a451e"


def main():
    """Copy a verified cached source and expose its existing comparison API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crate-source", required=True, type=Path, help="Cached rten-gemm-0.26.0 directory")
    parser.add_argument("--output", required=True, type=Path, help="New isolated probe directory")
    args = parser.parse_args()
    source = args.crate_source.resolve()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("output directory must not exist")
    vcs = json.loads((source / ".cargo_vcs_info.json").read_text())
    if vcs["git"]["sha1"] != REVISION:
        parser.error("unexpected upstream RTen revision")
    archive = source.parents[2] / "cache" / source.parent.name / "rten-gemm-0.26.0.crate"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        parser.error("cached archive differs from Cargo.lock's registry checksum")
    checksums = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = Path(*Path(member.name).parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                parser.error("unexpected archive member path")
            expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
            if actual != expected:
                parser.error(f"cached crate file differs from registry: {relative}")
            checksums[str(relative)] = expected
    destination.mkdir(parents=True)
    copied = destination / "rten-gemm"
    shutil.copytree(source, copied)
    library = copied / "src/lib.rs"
    contents = library.read_text()
    for declaration in ("enum F32KernelType", "enum Int8KernelType", "trait WithKernel"):
        if contents.count(f"\n{declaration}") != 1:
            raise ValueError(f"unexpected source declaration: {declaration}")
        contents = contents.replace(f"\n{declaration}", f"\npub {declaration}")
    library.write_text(contents)
    (destination / "src").mkdir()
    shutil.copyfile(Path(__file__).with_name("main.rs"), destination / "src/main.rs")
    shutil.copyfile(Path(__file__).with_name("Cargo.lock"), destination / "Cargo.lock")
    (destination / "Cargo.toml").write_text(
        '[package]\nname = "ocracle-gemm-probe"\nversion = "0.0.0"\nedition = "2024"\n\n'
        '[dependencies]\nrten-gemm = { path = "rten-gemm" }\nrten-tensor = "=0.26.0"\n'
    )
    (destination / "provenance.json").write_text(json.dumps({
        "upstream_revision": REVISION,
        "crate_version": "0.26.0",
        "crate_archive_sha256": ARCHIVE_SHA256,
        "source_change": "pub visibility for F32KernelType, Int8KernelType and WithKernel only",
        "source_library_sha256": checksums["src/lib.rs"],
        "copied_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256((destination / "src/main.rs").read_bytes()).hexdigest(),
        "scope": "synthetic float32 GEMM; not full OCR, scalar-only, or Android evidence",
    }, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
