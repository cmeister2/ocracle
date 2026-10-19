"""Pinned EasyOCR checkpoint acquisition, alphabet audits and ONNX adaptation.

Experimental exports are not registered production models. A successful export
proves compatibility, not coverage of a language, typeface or card treatment.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import unicodedata
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
SOURCES_PATH = HERE / "model_sources.json"
PROFILE_MODELS = {
    "ja": ["japanese_g1", "japanese_g2"],
    "latin": ["latin_g2"],
    "zhs": ["zh_sim_g2"],
    "zht": ["zh_tra_g1"],
    "ko": ["korean_g2"],
    "ru": ["cyrillic_g2"],
    "ar": ["arabic_g1"],
    "sa": ["devanagari_g1"],
}


def digest(data: bytes) -> str:
    """Return a SHA-256 artifact identity."""
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    """Download pinned public source or weights with a descriptive identity."""
    request = urllib.request.Request(url, headers={"User-Agent": "ocracle-easyocr-experiment/0.1 (+https://github.com/cmeister2/ocracle)"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def catalog() -> dict:
    """Read the checked-in upstream revision and checkpoint metadata."""
    return json.loads(SOURCES_PATH.read_text())


def reference_text(card: dict) -> str:
    """Read localized printing fields; never replace them with English Oracle text."""
    return "\n".join(
        printing.get(field, "") or ""
        for printing in [card, *card.get("card_faces", [])]
        for field in ["printed_name", "printed_text", "flavor_text"]
    )


def audit_corpus(corpus_path: Path, profile: str) -> dict:
    """Audit actual localized printing characters against each pretrained alphabet."""
    corpus = json.loads(corpus_path.read_text())
    cards = corpus["cards"] if isinstance(corpus, dict) else corpus
    selected = [card for card in cards if card["lang"] in ({"en", "de", "fr", "it", "es", "pt", "la"} if profile == "latin" else {profile})]
    if not selected:
        raise ValueError(f"No localized {profile!r} printings in {corpus_path}")
    characters: Counter[str] = Counter()
    appearances: dict[str, list[str]] = {}
    for card in selected:
        for character in unicodedata.normalize("NFC", reference_text(card)):
            if not character.isspace():
                characters[character] += 1
                examples = appearances.setdefault(character, [])
                if card["id"] not in examples and len(examples) < 3:
                    examples.append(card["id"])
    models = catalog()["models"]
    candidates = []
    for name in PROFILE_MODELS[profile]:
        alphabet = models[name]["characters"]
        missing = sorted(set(characters) - set(alphabet), key=ord)
        candidates.append({
            "model": name,
            "alphabet_characters": len(alphabet),
            "missing_unique_characters": len(missing),
            "missing_occurrences": sum(characters[character] for character in missing),
            "missing": [{"character": character, "codepoint": f"U+{ord(character):04X}",
                         "name": unicodedata.name(character, "UNKNOWN"), "occurrences": characters[character],
                         "example_print_ids": appearances[character]} for character in missing],
        })
    preferred = min(candidates, key=lambda entry: (entry["missing_occurrences"], entry["missing_unique_characters"], entry["alphabet_characters"]))
    return {
        "profile": profile,
        "corpus_sha256": digest(corpus_path.read_bytes()),
        "printings": len(selected),
        "unique_reference_characters": len(characters),
        "normalization": "NFC; whitespace excluded from the coverage count; fullwidth digits and punctuation preserved",
        "scope": "Alphabet membership only. Presence of a character does not prove that its printed glyph is recognized. Mana-symbol tokens in printing metadata are not verified image recognition.",
        "candidates": candidates,
        "preferred_model_by_coverage": preferred["model"],
        "complete_alphabet": preferred["missing_unique_characters"] == 0,
    }


def prepare_sources(cache: Path) -> Path:
    """Fetch digest-verified recognition source and remove unused detector imports."""
    metadata = catalog()
    revision = metadata["source_revision"]
    source_root = cache / "sources" / revision
    for name, expected_hash in metadata["source_files"].items():
        original = source_root / (name + ".original")
        original.parent.mkdir(parents=True, exist_ok=True)
        if not original.exists() or digest(original.read_bytes()) != expected_hash:
            data = download(f"https://raw.githubusercontent.com/JaidedAI/EasyOCR/{revision}/easyocr/{name}")
            if digest(data) != expected_hash:
                raise ValueError(f"Upstream source digest changed: {name}")
            original.write_bytes(data)
        source = original.read_text()
        if name == "model/modules.py":
            # These imports are used only by the dormant CRAFT detector class.
            source = source.replace("import torchvision\n", "").replace("from torchvision import models\n", "")
        destination = source_root / name
        destination.write_text(source)
    (source_root / "model/__init__.py").write_text("")
    return source_root


def checkpoint(name: str, cache: Path) -> tuple[Path, dict]:
    """Acquire official pretrained weights, checking upstream MD5 and pinned SHA-256."""
    entry = catalog()["models"][name]
    destination = cache / "checkpoints" / entry["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or hashlib.md5(destination.read_bytes()).hexdigest() != entry["md5sum"]:
        archive = download(entry["url"])
        with zipfile.ZipFile(io.BytesIO(archive)) as contents:
            data = contents.read(entry["filename"])
        if hashlib.md5(data).hexdigest() != entry["md5sum"]:
            raise ValueError(f"Official checkpoint MD5 mismatch: {name}")
        temporary = destination.with_suffix(".pth.part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    actual_hash = digest(destination.read_bytes())
    if entry.get("sha256") and actual_hash != entry["sha256"]:
        raise ValueError(f"Pinned checkpoint SHA-256 mismatch: {name}")
    return destination, {**entry, "observed_sha256": actual_hash, "bytes": destination.stat().st_size}


def load_network(name: str, cache: Path):
    """Restore the complete pretrained feature extractor, BiLSTMs and classifier."""
    import torch

    path, entry = checkpoint(name, cache)
    source_root = prepare_sources(cache)
    sys.path.insert(0, str(source_root))
    model_module = "model.vgg_model" if entry["generation"] == "gen2" else "model.model"
    model_type = importlib.import_module(model_module).Model
    channels = 256 if entry["generation"] == "gen2" else 512
    model = model_type(input_channel=1, output_channel=channels, hidden_size=channels,
                       num_class=len(entry["characters"]) + 1)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in weights.items()})
    model.eval()
    return model, entry


def compatible_wrapper(model):
    """Adapt upstream EasyOCR exactly to ocracle's grayscale/TNC log-probability ABI."""
    import torch
    from torch import nn

    class OcracleWrapper(nn.Module):
        """Keep learned layers and replace equivalent pooling with an ONNX mean."""

        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            # Existing Rust input is pixel/255-.5. EasyOCR expects [-1,1].
            visual = self.network.FeatureExtraction(image * 2)
            # Exactly upstream permute→AdaptiveAvgPool2d((None,1))→squeeze.
            sequence = visual.mean(2).permute(0, 2, 1)
            contextual = self.network.SequenceModeling(sequence)
            logits = self.network.Prediction(contextual.contiguous())
            return torch.log_softmax(logits.permute(1, 0, 2), dim=2)

    return OcracleWrapper(model).eval()


def export_model(name: str, cache: Path, output: Path, language: str, audit: dict | None) -> dict:
    """Export full pretrained weights and verify dynamic-width ONNX numerical parity."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(1993)
    model, entry = load_network(name, cache)
    wrapper = compatible_wrapper(model)
    example = torch.rand(1, 1, 64, 160) - 0.5
    with torch.inference_mode():
        expected = torch.log_softmax(model(example * 2, None).permute(1, 0, 2), dim=2)
        wrapped = wrapper(example)
        wrapper_error = float((expected - wrapped).abs().max())
    if wrapper_error > 1e-5:
        raise ValueError(f"Adapted pooling changed upstream output: {wrapper_error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, example, output, input_names=["image"], output_names=["log_probs"],
                      dynamic_axes={"image": {3: "width"}, "log_probs": {0: "time"}},
                      opset_version=17, dynamo=False)
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    parity = []
    for width in [128, 160, 287]:
        sample = torch.rand(1, 1, 64, width) - 0.5
        with torch.inference_mode():
            reference = wrapper(sample).numpy()
        observed = session.run(None, {"image": sample.numpy()})[0]
        maximum_error = float(np.max(np.abs(reference - observed)))
        same_decode = bool(np.array_equal(reference.argmax(2), observed.argmax(2)))
        if maximum_error > 2e-4 or not same_decode:
            raise ValueError(f"ONNX parity failed at width {width}: {maximum_error}")
        parity.append({"width": width, "output_shape": list(observed.shape),
                       "max_absolute_log_probability_error": maximum_error, "identical_argmax": same_decode})
    backbone = "VGG" if entry["generation"] == "gen2" else "ResNet"
    config = {
        "alphabet": entry["characters"], "input_height": 64,
        "model_name": f"easyocr-{name}-pretrained-experiment", "language": language,
        "architecture": f"EasyOCR CRNN ({backbone} + two BiLSTMs)",
        "model_method": f"EasyOCR CRNN ({backbone} + two BiLSTMs) / greedy CTC",
        "normalization_adapter": "ONNX multiplies current Rust pixel/255-.5 input by 2 to match EasyOCR [-1,1]",
        "resize_note": "Upstream EasyOCR uses ceiling width and bicubic resizing; current Rust uses ties-even rounding and triangle resizing. End-to-end print output must be evaluated separately.",
        "experimental": True,
    }
    output.with_suffix(".json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "checkpoint": name,
        "upstream_repository": catalog()["source_repository"], "source_revision": entry["source_revision"],
        "checkpoint_url": entry["url"], "checkpoint_sha256": entry["observed_sha256"],
        "license": entry["license"], "source_changes": "Removed unused torchvision detector imports; equivalent height pooling uses mean; input scaling and TNC log_softmax wrapper added.",
        "onnx_sha256": digest(output.read_bytes()), "onnx_bytes": output.stat().st_size,
        "onnx_operators": sorted({node.op_type for node in graph.graph.node}),
        "wrapper_max_error_against_upstream": wrapper_error, "pytorch_onnx_parity": parity,
        "alphabet_audit": audit,
        "limitations": "Unmodified pretrained recognition, not MTG-fine-tuned. Alphabet gaps are explicit in the audit. No language-wide, font-wide, vertical text or furigana coverage claim.",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return provenance
