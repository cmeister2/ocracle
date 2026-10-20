#!/usr/bin/env python3
"""Render paired measured latency/CER charts and original-coordinate OCR overlays.

Requires matplotlib and Pillow. Inputs are performance_suite.py JSON artifacts;
all runs must contain exactly the same fixtures. Example:
  python scripts/performance_report.py --run before.json --run after.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from performance_suite import ROOT, check_inference, score, sha256, write_json

COLORS = ("#4b6b88", "#007f6d", "#b87318", "#9863a8")
WORKLOADS = ("line", "multiline", "card")
NAMES = {"line": "Single line", "multiline": "Multiline body", "card": "Whole card"}


def validate(runs: list[dict]) -> list[str]:
    """Require complete, identically paired workloads and independently score output."""
    first = runs[0]
    ids = [record["sample"]["id"] for record in first["records"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    reference = {record["sample"]["id"]: record["sample"] for record in first["records"]}
    reference_layouts = {
        record["sample"]["id"]: record.get("effective_layout", record["inference"].get("layout"))
        for record in first["records"]
    }
    for run in runs:
        if run["status"] != "complete":
            raise ValueError(f"Incomplete benchmark run: {run['label']}")
        records = {record["sample"]["id"]: record for record in run["records"]}
        if len(records) != len(run["records"]) or set(records) != set(ids):
            raise ValueError("Runs must contain exactly the same sample IDs")
        if len(run["requested_samples"]) != len(records) or set(run["requested_samples"]) != set(ids):
            raise ValueError("Completed run does not contain exactly the requested samples")
        for identifier in ids:
            record = records[identifier]
            if record["sample"] != reference[identifier]:
                raise ValueError(f"Fixture or scoring protocol changed: {identifier}")
            layout = record.get("effective_layout", record["inference"].get("layout"))
            if layout is None or layout != reference_layouts[identifier]:
                raise ValueError(f"Paired layout changed or is missing: {identifier}")
            if record["inference"].get("layout") != layout:
                raise ValueError(f"Recorded layout differs from actual inference: {identifier}")
            check_inference(record["sample"], record["inference"], run["iterations"])
            if record["evaluation"] != score(record["sample"], record["inference"]):
                raise ValueError(f"Stored evaluation does not match actual text: {identifier}")
        if (run["threads"], run["host"]) != (first["threads"], first["host"]):
            raise ValueError("Paired comparisons require the same thread count and host")
    if len({run["label"] for run in runs}) != len(runs):
        raise ValueError("Run labels must be distinct")
    return ids


def geomean(values: list[float]) -> float:
    """Give each image equal multiplicative weight in the latency summary."""
    return math.exp(sum(math.log(value) for value in values) / len(values))


def title_only(runs: list[dict]) -> bool:
    """Identify a nonempty comparison consisting entirely of English title crops."""
    return bool(runs and runs[0]["records"]) and all(
        record["sample"].get("language") == "en"
        and record["sample"]["workload"] == "line"
        and "title" in record["sample"]["id"]
        for run in runs for record in run["records"]
    )


def summaries(runs: list[dict], identifiers: list[str]) -> dict:
    """Retain every per-image change as well as workload-level summaries."""
    baseline = {record["sample"]["id"]: record for record in runs[0]["records"]}
    result = {"baseline": runs[0]["label"], "runs": []}
    for run in runs:
        records = {record["sample"]["id"]: record for record in run["records"]}
        groups = {}
        for workload in WORKLOADS:
            group = [record for record in records.values()
                     if record["sample"]["workload"] == workload]
            if not group:
                continue
            edits = sum(record["evaluation"]["character_edits"] for record in group)
            characters = sum(record["evaluation"]["reference_characters"] for record in group)
            groups[workload] = {
                "cases": len(group),
                "geomean_case_median_ms": geomean(
                    [record["inference"]["median_ms"] for record in group]),
                "worst_case_p95_ms": max(record["inference"]["p95_ms"] for record in group),
                "character_edits": edits, "reference_characters": characters,
                "character_error_rate": edits / max(characters, 1),
                "exact_matches": sum(record["evaluation"]["exact_match"] for record in group),
            }
        paired = []
        for identifier in identifiers:
            before, after = baseline[identifier], records[identifier]
            paired.append({
                "sample_id": identifier,
                "speedup": before["inference"]["median_ms"] / after["inference"]["median_ms"],
                "character_edits_before": before["evaluation"]["character_edits"],
                "character_edits_after": after["evaluation"]["character_edits"],
                "text_changed": before["inference"]["text"] != after["inference"]["text"],
                "median_ms": after["inference"]["median_ms"],
                "p95_ms": after["inference"]["p95_ms"],
            })
        result["runs"].append({"label": run["label"], "groups": groups, "paired": paired})
    return result


def charts(summary: dict, output: Path, threads: int, title_crops: bool = False) -> None:
    """Plot desktop latency and retained transcription errors with explicit scope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    groups = [workload for workload in WORKLOADS
              if workload in summary["runs"][0]["groups"]]
    figure, axes = plt.subplots(len(groups), 2, squeeze=False,
                                figsize=(13, 5.2 if len(groups) == 1 else 3.0 * len(groups) + 1.6))
    # Reserve room for the two-row legend while retaining useful plot height.
    figure.subplots_adjust(top=0.74 if len(groups) == 1 else 0.86,
                           bottom=0.17 if len(groups) == 1 else 0.10,
                           hspace=0.5, wspace=0.25, left=0.075, right=0.98)
    for group_index, group in enumerate(groups):
        times, rates = [], []
        left, right = axes[group_index]
        for index, run in enumerate(summary["runs"]):
            time_ms = run["groups"][group]["geomean_case_median_ms"]
            rate = 100 * run["groups"][group]["character_error_rate"]
            times.append(time_ms)
            rates.append(rate)
            for axis, value, unit in ((left, time_ms, " ms"), (right, rate, "%")):
                bars = axis.bar(index, value, width=0.62, color=COLORS[index], label=run["label"])
                axis.bar_label(bars, labels=[f"{value:.1f}{unit}"], padding=4, fontsize=11)
        left.axhline(1000 / 15, color="#b3333a", linestyle="--", linewidth=1.5,
                     label="66.7 ms target budget")
        left.set_ylim(0, max(max(times), 1000 / 15) * 1.28)
        right.set_ylim(0, max(max(rates) * 1.28, 0.5))
        left.set_ylabel("Warm OCR latency (ms)")
        right.set_ylabel("Character error rate (%)")
        cases = summary["runs"][0]["groups"][group]["cases"]
        name = "Title crop" if title_crops and group == "line" else NAMES[group]
        title = f"{name} · {cases} {'case' if cases == 1 else 'cases'}"
        left.set_title(f"{title} · latency", loc="left")
        right.set_title(f"{title} · accuracy", loc="left")
        for axis in (left, right):
            axis.set_xticks([])
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                  ncols=min(3, len(handles)), frameon=False, fontsize=10)
    figure.suptitle(f"Same images, {threads} CPU threads • desktop results, NOT Android FPS",
                    fontsize=15, weight="bold", y=0.99)
    if title_crops:
        footer = (
            "Latency: geometric mean of per-title warm medians. All clean and blurred crops retained.\n"
            "Detection, title localization, image decode/model load excluded. "
            "No Android handset, camera or thermal measurements."
        )
    else:
        footer = (
            "Latency: geometric mean of per-case warm medians; axes differ by workload. "
            "Image decode/model load excluded.\n"
            "Whole-card CER: fixed title/type/body regions only. No Android handset, camera "
            "or thermal measurements."
        )
    figure.text(0.5, 0.02, footer, ha="center", fontsize=10)
    figure.savefig(output / "latency-accuracy.png", dpi=160)
    plt.close(figure)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Unicode-capable font present on ordinary Linux development hosts."""
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


def pixel_wrap(text: str, face: ImageFont.ImageFont, width: int) -> list[str]:
    """Wrap display text without deleting punctuation, spaces or other characters."""
    lines, current = [], ""
    for character in text:
        if current and face.getlength(current + character) > width:
            lines.append(current)
            current = ""
        current += character
    return lines + [current]


def title_grid(runs: list[dict], identifiers: list[str], output: Path) -> None:
    """Show every measured title crop, literal strings, timing and scored errors."""
    if not title_only(runs):
        raise ValueError("Title grid requires English title-only comparisons")
    records = [{record["sample"]["id"]: record for record in run["records"]}
               for run in runs]
    heading_face, reference_face, label_face, prediction_face = (
        font(22), font(21), font(16), font(21)
    )
    panel_width, padding, crop_height, gap = 930, 24, 104, 18
    content_width = panel_width - 2 * padding
    panels = []
    for identifier in identifiers:
        sample = records[0][identifier]["sample"]
        path = ROOT / sample["image"]
        if sha256(path) != sample["image_sha256"]:
            raise ValueError(f"Title grid image differs from measured input: {path}")
        with Image.open(path) as source:
            original = source.convert("RGB")
        heading = pixel_wrap(
            f"{identifier}  ·  {original.width}×{original.height} pixels",
            heading_face, content_width,
        )
        reference = pixel_wrap(
            "Reference: " + json.dumps(sample["ground_truth"], ensure_ascii=False),
            reference_face, content_width,
        )
        blocks = []
        for run, mapped in zip(runs, records):
            record = mapped[identifier]
            edits = record["evaluation"]["character_edits"]
            status = f"ERROR: {edits} character edit{'s' if edits != 1 else ''}" if edits else "0 edits"
            label = pixel_wrap(
                f"{run['label']}  ·  {record['inference']['median_ms']:.1f} ms median  ·  {status}",
                label_face, content_width - 24,
            )
            prediction = pixel_wrap(
                json.dumps(record["inference"]["text"], ensure_ascii=False),
                prediction_face, content_width - 24,
            )
            height = 14 + 21 * len(label) + 27 * len(prediction)
            blocks.append((label, prediction, height, bool(edits)))
        crop_top = padding + 29 * len(heading) + 10
        reference_top = crop_top + crop_height + 15
        blocks_top = reference_top + 28 * len(reference) + 12
        panel_height = blocks_top + sum(block[2] + 8 for block in blocks) + padding
        panel = Image.new("RGB", (panel_width, panel_height), "#f7f8fa")
        draw = ImageDraw.Draw(panel)
        for row, line in enumerate(heading):
            draw.text((padding, padding + 29 * row), line, font=heading_face, fill="#192a38")
        draw.rectangle(
            (padding, crop_top, panel_width - padding, crop_top + crop_height),
            fill="white", outline="#d7dee4",
        )
        scale = min((content_width - 8) / original.width, (crop_height - 8) / original.height)
        size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
        # Nearest-neighbor enlargement keeps the blur/degradation visible.
        resampling = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS
        panel.paste(original.resize(size, resampling),
                    (padding + (content_width - size[0]) // 2,
                     crop_top + (crop_height - size[1]) // 2))
        for row, line in enumerate(reference):
            draw.text((padding, reference_top + 28 * row), line,
                      font=reference_face, fill="#192a38")
        top = blocks_top
        for index, (label, prediction, height, error) in enumerate(blocks):
            draw.rectangle((padding, top, panel_width - padding, top + height),
                           fill="#fff0ee" if error else "#edf6f1")
            draw.rectangle((padding, top, padding + 4, top + height), fill=COLORS[index])
            color = "#9f2525" if error else "#184d39"
            for row, line in enumerate(label):
                draw.text((padding + 12, top + 5 + 21 * row), line,
                          font=label_face, fill=color)
            prediction_top = top + 7 + 21 * len(label)
            for row, line in enumerate(prediction):
                draw.text((padding + 12, prediction_top + 27 * row), line,
                          font=prediction_face, fill=color)
            top += height + 8
        panels.append(panel)

    columns = min(2, len(panels))
    row_heights = [max(panel.height for panel in panels[start:start + columns])
                   for start in range(0, len(panels), columns)]
    header_height = 122
    width = columns * panel_width + (columns + 1) * gap
    footer_face = font(17)
    footer_lines = []
    for line in (
        "Crops scaled only for display. Error counts use NFC + collapsed whitespace; case and punctuation retained.",
        "Title localization, detection, image decode and model load excluded. These are not Android FPS measurements.",
        "Card imagery © Wizards of the Coast; scans via Scryfall.",
    ):
        footer_lines.extend(pixel_wrap(line, footer_face, width - 2 * gap))
    footer_height = 24 + 28 * len(footer_lines)
    height = header_height + sum(row_heights) + len(row_heights) * gap + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 15), f"Title crop OCR · all {len(identifiers)} measured inputs",
              font=font(30), fill="#192a38")
    draw.text((gap, 58), f"{runs[0]['threads']} CPU threads · desktop warm medians · "
              "clean and blurred variants retained", font=font(20), fill="#334f5f")
    draw.text((gap, 88), "Literal reference/prediction strings are quoted; "
              "red rows mark transcription errors.", font=font(18), fill="#334f5f")
    top = header_height
    for row_index, row_height in enumerate(row_heights):
        for column, panel in enumerate(panels[row_index * columns:(row_index + 1) * columns]):
            canvas.paste(panel, (gap + column * (panel_width + gap), top))
        top += row_height + gap
    for row, line in enumerate(footer_lines):
        draw.text((gap, top + 10 + row * 28), line, font=footer_face, fill="#334f5f")
    canvas.save(output / "title-grid.png")


def overlay(runs: list[dict], identifier: str, output: Path) -> None:
    """Draw measured line boxes in the original image coordinate system."""
    records = [next(record for record in run["records"] if record["sample"]["id"] == identifier)
               for run in runs]
    sample = records[0]["sample"]
    path = ROOT / sample["image"]
    if sha256(path) != sample["image_sha256"]:
        raise ValueError(f"Overlay image differs from measured input: {path}")
    original = Image.open(path).convert("RGB")
    scale = min(1.0, 450 / original.width, 610 / original.height)
    image_width, image_height = round(original.width * scale), round(original.height * scale)
    panels = []
    for index, (run, record) in enumerate(zip(runs, records)):
        lines = record["inference"]["lines"]
        wrapped = []
        for number, line in enumerate(lines, 1):
            wrapped.extend(textwrap.wrap(f"{number:02d}  {line['text']}", width=64) or [""])
        row_height = max(image_height + 105, 130 + 22 * len(wrapped))
        panel = Image.new("RGB", (1100, row_height), "#f7f8fa")
        draw = ImageDraw.Draw(panel)
        timing = record["inference"]
        evaluation = record["evaluation"]
        draw.text((20, 15), f"{run['label']}  •  {identifier}", font=font(22), fill="#192a38")
        draw.text((20, 48), f"Median {timing['median_ms']:.1f} ms  /  "
                  f"p95 {timing['p95_ms']:.1f} ms  •  "
                  f"CER {evaluation['character_error_rate']:.1%} "
                  f"({evaluation['character_edits']}/{evaluation['reference_characters']})",
                  font=font(17), fill="#334f5f")
        panel.paste(original.resize((image_width, image_height), Image.Resampling.LANCZOS),
                    (20, 85))
        draw = ImageDraw.Draw(panel)
        for number, line in enumerate(lines, 1):
            left, top, width, height = line["bounds"]
            bounds = [20 + left * scale, 85 + top * scale,
                      20 + (left + width) * scale, 85 + (top + height) * scale]
            draw.rectangle(bounds, outline=COLORS[index], width=2)
            marker = f"{number}"
            box = draw.textbbox((bounds[0], bounds[1]), marker, font=font(13))
            draw.rectangle(box, fill=COLORS[index])
            draw.text((bounds[0], bounds[1]), marker, font=font(13), fill="white")
        for line_index, line in enumerate(wrapped):
            draw.text((495, 90 + 22 * line_index), line, font=font(16), fill="#192a38")
        panels.append(panel)
    canvas = Image.new("RGB", (1100, sum(panel.height for panel in panels) + 80), "white")
    top = 0
    for panel in panels:
        canvas.paste(panel, (0, top))
        top += panel.height
    draw = ImageDraw.Draw(canvas)
    draw.text((20, top + 12), "Boxes are actual OCR results in original image coordinates, "
              "scaled only for display.", font=font(16), fill="#334f5f")
    draw.text((20, top + 39), "Desktop latency is not Android FPS. "
              "Whole-card CER scores fixed title/type/body regions; other lines remain visible.",
              font=font(14), fill="#334f5f")
    canvas.save(output / f"{identifier}-overlay.png", optimize=True)


def main() -> None:
    """Validate paired artifacts before creating charts, summaries and overlays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/performance/results")
    parser.add_argument("--overlay", action="append", help="Sample ID; repeat to select overlays")
    args = parser.parse_args()
    if not 1 <= len(args.run) <= len(COLORS):
        parser.error("Supply one to four comparison runs")
    runs = [json.loads(path.read_text()) for path in args.run]
    identifiers = validate(runs)
    summary = summaries(runs, identifiers)
    summary["inputs"] = [{"path": str(path), "sha256": sha256(path)} for path in args.run]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", summary)
    titles = title_only(runs)
    charts(summary, args.output, runs[0]["threads"], title_crops=titles)
    if titles:
        title_grid(runs, identifiers, args.output)
    selected = args.overlay or [identifier for identifier in
                                ("card-divination", "multiline-divination")
                                if identifier in identifiers]
    for identifier in selected:
        if identifier not in identifiers:
            parser.error(f"Overlay sample was not measured: {identifier}")
        overlay(runs, identifier, args.output)
    print(f"Wrote paired charts, summary, {len(selected)} overlays"
          f"{' and title-grid.png' if titles else ''} to {args.output}")


if __name__ == "__main__":
    main()
