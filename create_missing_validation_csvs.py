from pathlib import Path
import csv

out = Path("validation_data")
out.mkdir(exist_ok=True)

header = [
    "source_id",
    "source_type",
    "tool_version",
    "configuration_hash",
    "date",
    "provenance_reference",
    "independent",
    "strategy",
    "spans",
    "band",
    "wavelength_nm",
    "metric",
    "metric_unit",
    "reference_value",
    "reference_uncertainty",
    "notes",
]

calibration_rows = [
    ("gnpy_calibration_flat_1span_1535nm", 1, 1535.000023, 28.53),
    ("gnpy_calibration_flat_1span_1550nm", 1, 1549.999992, 28.58),
    ("gnpy_calibration_flat_1span_1560nm", 1, 1560.000021, 28.62),
    ("gnpy_calibration_flat_4span_1535nm", 4, 1535.000023, 22.82),
    ("gnpy_calibration_flat_4span_1550nm", 4, 1549.999992, 22.87),
    ("gnpy_calibration_flat_4span_1560nm", 4, 1560.000021, 22.91),
    ("gnpy_calibration_flat_8span_1535nm", 8, 1535.000023, 19.85),
    ("gnpy_calibration_flat_8span_1550nm", 8, 1549.999992, 19.90),
    ("gnpy_calibration_flat_8span_1560nm", 8, 1560.000021, 19.94),
]

holdout_rows = [
    ("gnpy_holdout_flat_1span_1540nm", 1, 1539.802234, 28.28),
    ("gnpy_holdout_flat_1span_1545nm", 1, 1545.199049, 28.30),
    ("gnpy_holdout_flat_1span_1555nm", 1, 1554.798364, 28.36),
    ("gnpy_holdout_flat_4span_1540nm", 4, 1539.802234, 22.62),
    ("gnpy_holdout_flat_4span_1545nm", 4, 1545.199049, 22.64),
    ("gnpy_holdout_flat_4span_1555nm", 4, 1554.798364, 22.70),
    ("gnpy_holdout_flat_8span_1540nm", 8, 1539.802234, 19.67),
    ("gnpy_holdout_flat_8span_1545nm", 8, 1545.199049, 19.68),
    ("gnpy_holdout_flat_8span_1555nm", 8, 1554.798364, 19.74),
]

def write_reference_csv(path: Path, rows, provenance: str):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for source_id, spans, wavelength_nm, reference_value in rows:
            writer.writerow([
                source_id,
                "GNPy",
                "2.14.1",
                "final_heldout_validation_reproducibility_snapshot",
                "2026-07-22",
                provenance,
                "true",
                "flat",
                spans,
                "C",
                f"{wavelength_nm:.6f}",
                "gsnr_db",
                "dB",
                f"{reference_value:.2f}",
                "",
                "Reconstructed from final validated GNPy evidence used in the manuscript; simulation-to-simulation reference, not experimental measurement.",
            ])

write_reference_csv(
    out / "gnpy_calibration_train_1535_1550_1560.csv",
    calibration_rows,
    "final_calibration_reference_rows_1535_1550_1560",
)

write_reference_csv(
    out / "gnpy_holdout_1540_1545_1555.csv",
    holdout_rows,
    "final_holdout_reference_rows_1540_1545_1555",
)

print("DONE")
print("Created:", out / "gnpy_calibration_train_1535_1550_1560.csv")
print("Created:", out / "gnpy_holdout_1540_1545_1555.csv")
