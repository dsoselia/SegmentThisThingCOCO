#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stt_pipeline.data import IndexedJsonl


def main() -> None:
    parser = argparse.ArgumentParser("Build .offsets.npy sidecars for JSONL manifests.")
    parser.add_argument("manifest", nargs="+")
    args = parser.parse_args()

    summaries = []
    for manifest in args.manifest:
        indexed = IndexedJsonl(manifest)
        sidecar = indexed.write_sidecar()
        summaries.append(
            {
                "manifest": str(Path(manifest)),
                "sidecar": str(sidecar),
                "entries": len(indexed),
                "sidecar_bytes": sidecar.stat().st_size,
            }
        )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
