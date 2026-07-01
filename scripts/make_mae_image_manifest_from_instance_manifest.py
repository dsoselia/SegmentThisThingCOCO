#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Create an image-only MAE manifest from an instance/RLE manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dedupe", action="store_true", help="Drop repeated image_path entries while preserving first occurrence.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    seen: set[str] = set()
    read_count = 0
    write_count = 0

    with input_path.open() as source, tmp_path.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            read_count += 1
            entry = json.loads(line)
            image_path = entry["image_path"]
            if args.dedupe:
                if image_path in seen:
                    continue
                seen.add(image_path)
            target.write(
                json.dumps(
                    {
                        "image_path": image_path,
                        "dataset_name": entry.get("dataset_name", "sa1b_subset"),
                    }
                )
                + "\n"
            )
            write_count += 1

    tmp_path.replace(output_path)
    print(json.dumps({"input": str(input_path), "output": str(output_path), "read": read_count, "written": write_count}, indent=2))


if __name__ == "__main__":
    main()
