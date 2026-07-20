import pandas as pd
from pathlib import Path

model_path = Path("runs/q3-heldout-model-001/results/channel_performance.csv")
cal_path = Path("validation_data/gnpy_calibration_train_1535_1550_1560.csv")
hold_path = Path("validation_data/gnpy_holdout_1540_1545_1555.csv")

model = pd.read_csv(model_path)
cal = pd.read_csv(cal_path)
hold = pd.read_csv(hold_path)

print("MODEL rows:", len(model))
print("MODEL columns:", list(model.columns))
print("\nMODEL strategies:", sorted(model["strategy"].astype(str).unique()) if "strategy" in model else "NO strategy column")
print("MODEL spans:", sorted(pd.to_numeric(model["spans"], errors="coerce").dropna().astype(int).unique()) if "spans" in model else "NO spans column")
print("MODEL wavelength min/max:", model["wavelength_nm"].min(), model["wavelength_nm"].max())

def check_split(name, ext):
    print(f"\n=== {name} ===")
    matched = 0
    for _, row in ext.iterrows():
        strategy = str(row["strategy"]).strip().lower()
        spans = int(row["spans"])
        target = float(row["wavelength_nm"])

        sub = model[
            (model["strategy"].astype(str).str.strip().str.lower() == strategy)
            & (pd.to_numeric(model["spans"], errors="coerce").astype("Int64") == spans)
        ].copy()

        if sub.empty:
            print(f"NO MODEL SUBSET: strategy={strategy}, spans={spans}, target={target}")
            continue

        sub["err"] = (sub["wavelength_nm"].astype(float) - target).abs()
        nearest = sub.sort_values("err").iloc[0]
        ok = float(nearest["err"]) <= 0.5
        matched += int(ok)
        print(
            f"target={target:.6f}, spans={spans}, strategy={strategy} -> "
            f"nearest_model={float(nearest['wavelength_nm']):.6f}, "
            f"err={float(nearest['err']):.6f}, matched_0.5nm={ok}"
        )

    print(f"{name} matched within 0.5 nm:", matched, "of", len(ext))

check_split("CALIBRATION", cal)
check_split("HOLDOUT", hold)
