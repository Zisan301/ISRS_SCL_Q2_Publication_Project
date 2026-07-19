#!/usr/bin/env python
"""Build 1/4/8-span GNPy flat network JSON files from edfa_example_network.json.

Usage:
python tools\make_gnpy_flat_span_cases.py --example-data "C:\path\to\gnpy\example-data"
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example-data", required=True, help="Folder printed by gnpy.tools.cli_examples.show_example_data_dir")
    ap.add_argument("--out-dir", default="external_validation/gnpy/cases")
    ap.add_argument("--spans", default="1,4,8")
    args = ap.parse_args()

    base_path = Path(args.example_data) / "edfa_example_network.json"
    if not base_path.exists():
        raise SystemExit(f"Could not find: {base_path}")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    elements = base.get("elements")
    if not isinstance(elements, list):
        raise SystemExit("Unexpected GNPy network JSON: missing elements list")

    txs = [e for e in elements if str(e.get("type", "")).lower() == "transceiver"]
    fibers = [e for e in elements if str(e.get("type", "")).lower() == "fiber"]
    edfas = [e for e in elements if str(e.get("type", "")).lower() == "edfa"]

    if len(txs) < 2 or not fibers or not edfas:
        raise SystemExit("Base example must contain two transceivers, one fiber, and one EDFA")

    src = copy.deepcopy(txs[0])
    dst = copy.deepcopy(txs[-1])
    src["uid"] = "Site_A"
    dst["uid"] = "Site_B"
    fiber_template = fibers[0]
    edfa_template = edfas[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n_spans in [int(x.strip()) for x in args.spans.split(",") if x.strip()]:
        new = copy.deepcopy(base)
        new_elements = [src]
        new_connections = []
        previous = "Site_A"

        for idx in range(1, n_spans + 1):
            fiber = copy.deepcopy(fiber_template)
            fiber["uid"] = f"Span{idx}"
            edfa = copy.deepcopy(edfa_template)
            edfa["uid"] = f"Edfa{idx}"

            new_elements.append(fiber)
            new_elements.append(edfa)
            new_connections.append({"from_node": previous, "to_node": fiber["uid"]})
            new_connections.append({"from_node": fiber["uid"], "to_node": edfa["uid"]})
            previous = edfa["uid"]

        new_elements.append(dst)
        new_connections.append({"from_node": previous, "to_node": "Site_B"})
        new["elements"] = new_elements
        new["connections"] = new_connections

        out_path = out_dir / f"gnpy_flat_{n_spans}span_network.json"
        out_path.write_text(json.dumps(new, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
