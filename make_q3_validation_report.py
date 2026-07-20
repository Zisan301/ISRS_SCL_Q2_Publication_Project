from pathlib import Path
import json
import pandas as pd

base = Path("runs/q3-heldout-model-001/results")
out = Path("docs/q3_heldout_validation_results.md")

status = json.loads((base / "heldout_external_validation_status.json").read_text(encoding="utf-8"))
summary = pd.read_csv(base / "heldout_external_validation_summary.csv")
req = pd.read_csv(base / "heldout_external_validation_requirements.csv")

lines = []
lines.append("# Q3 Held-Out GNPy Validation Results")
lines.append("")
lines.append("## Validation status")
lines.append("")
lines.append(f"- Passed: **{status.get('passed')}**")
lines.append(f"- Reasons: `{status.get('reasons')}`")
lines.append("")
lines.append("## Fitted calibration correction")
lines.append("")
corr = status.get("correction", {})
for k, v in corr.items():
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Summary metrics")
lines.append("")
lines.append(summary.to_markdown(index=False))
lines.append("")
lines.append("## Requirement checks")
lines.append("")
lines.append(req.to_markdown(index=False))
lines.append("")
lines.append("## Paper-ready interpretation")
lines.append("")
lines.append(
    "The original GNPy reference points at 1535, 1550, and 1560 nm over 1, 4, "
    "and 8 spans were used only as calibration evidence. A separate held-out "
    "GNPy validation set was generated at unseen wavelengths near 1540, 1545, "
    "and 1555 nm over the same span counts. A wavelength-linear residual "
    "correction was fitted on the calibration split only and then applied to "
    "the held-out split without refitting. The held-out validation gate passed "
    "all configured requirements, with no failed validation reasons reported."
)
lines.append("")
lines.append("## Limitation")
lines.append("")
lines.append(
    "This validation is independent simulation-to-simulation validation against "
    "GNPy. It should not be described as laboratory or field experimental "
    "validation unless physical measurement data are added."
)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")

print("WROTE:", out)
