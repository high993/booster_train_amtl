#!/usr/bin/env python3
"""Export a TensorBoard event file to a plain-text summary.

Usage:
    python scripts/tfevents_to_txt.py /path/to/events.out.tfevents... \
        --output /path/to/export.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a TensorBoard event file to text.")
    parser.add_argument("input_file", type=Path, help="Path to the TensorBoard event file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file path. Defaults to the input path with a .txt suffix.",
    )
    return parser.parse_args()


def default_output_path(input_file: Path) -> Path:
    return input_file.with_suffix(input_file.suffix + ".txt")


def load_tensorboard_modules():
    try:
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.util import tensor_util
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the 'tensorboard' package. Run it from the training environment or install it with "
            "'python -m pip install tensorboard'."
        ) from exc
    return event_accumulator, tensor_util


def format_scalar_section(accumulator, tag: str) -> list[str]:
    lines = [f"[scalar] {tag}"]
    for event in accumulator.Scalars(tag):
        lines.append(f"step={event.step} wall_time={event.wall_time:.6f} value={event.value}")
    lines.append("")
    return lines


def format_tensor_value(value: object) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return repr(value)


def format_tensor_section(accumulator, tensor_util, tag: str) -> list[str]:
    lines = [f"[tensor] {tag}"]
    for event in accumulator.Tensors(tag):
        array_value = tensor_util.make_ndarray(event.tensor_proto)
        lines.append(
            "step={step} wall_time={wall_time:.6f} dtype={dtype} shape={shape} value={value}".format(
                step=event.step,
                wall_time=event.wall_time,
                dtype=array_value.dtype,
                shape=tuple(array_value.shape),
                value=format_tensor_value(array_value),
            )
        )
    lines.append("")
    return lines


def export_event_file(input_file: Path, output_file: Path) -> None:
    event_accumulator, tensor_util = load_tensorboard_modules()
    size_guidance = {
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
    }
    accumulator = event_accumulator.EventAccumulator(str(input_file), size_guidance=size_guidance)
    accumulator.Reload()
    tags = accumulator.Tags()

    lines = [f"source={input_file}", ""]

    scalar_tags = sorted(tags.get("scalars", []))
    tensor_tags = sorted(tags.get("tensors", []))

    lines.append("[metadata]")
    lines.append(f"scalar_tags={len(scalar_tags)}")
    lines.append(f"tensor_tags={len(tensor_tags)}")
    lines.append("")

    for tag in scalar_tags:
        lines.extend(format_scalar_section(accumulator, tag))

    for tag in tensor_tags:
        lines.extend(format_tensor_section(accumulator, tensor_util, tag))

    output_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_file = (args.output or default_output_path(input_file)).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    export_event_file(input_file, output_file)
    print(f"Exported {input_file} to {output_file}")


if __name__ == "__main__":
    main()