"""Clean the generated shipment export and write Part A evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, build_features, clean_shipments


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "raw_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")
deduplicated = raw.drop_duplicates().copy()

service_spellings_before = int(deduplicated["service_type"].nunique())
missing_distance = int(deduplicated["distance_km"].isna().sum())
missing_weight = int(deduplicated["parcel_weight_kg"].isna().sum())
missing_value = int(deduplicated["declared_value"].isna().sum())
missing_payment = int(deduplicated["payment_mode"].isna().sum())
sentinel_weight = int((deduplicated["parcel_weight_kg"] > 100).sum())

clean, medians = clean_shipments(raw)
service_spellings_after = int(clean["service_type"].nunique())
modeling = build_features(clean, hubs)
rows_after_hub_merge = len(modeling)
modeling["is_delayed"] = (modeling["delivery_hours"] > modeling["promised_hours"]).astype("int64")
modeling.to_csv(DATA / "modelling_table.csv", index=False)

q1 = float(modeling["delivery_hours"].quantile(0.25))
q3 = float(modeling["delivery_hours"].quantile(0.75))
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
log_value = np.log(modeling["declared_value_inr"])
checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(deduplicated)),
    ("rows_after_dedup", len(deduplicated)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_weight),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", rows_after_hub_merge),
    ("nulls_in_features_and_target", int(modeling[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(modeling["is_delayed"].sum())),
    ("delay_rate", modeling["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((modeling["delivery_hours"] < lower) | (modeling["delivery_hours"] > upper)).sum())),
    ("declared_value_mean_inr", modeling["declared_value_inr"].mean()),
    ("declared_value_median_inr", modeling["declared_value_inr"].median()),
    ("log_declared_value_mean", log_value.mean()),
    ("log_declared_value_median", log_value.median()),
]
pd.DataFrame(checks, columns=["metric", "value"]).to_csv(OUTPUTS / "part_a_checks.csv", index=False, float_format="%.4f")

distance_labels = ["0-200", "201-400", "401-600", "601+"]
modeling["distance_band"] = pd.cut(
    modeling["distance_km"], bins=[0, 200, 400, 600, 2000], labels=distance_labels, right=True,
)
group_rows = []
for dimension, column in [("service_type", "service_type"), ("hub_tier", "hub_tier"), ("distance_band", "distance_band")]:
    for group, frame in modeling.groupby(column, observed=False, sort=True):
        if frame.empty:
            continue
        group_rows.append({
            "dimension": dimension,
            "group": group,
            "shipments": len(frame),
            "delayed": int(frame["is_delayed"].sum()),
            "delay_rate": frame["is_delayed"].mean(),
            "mean_hours": frame["delivery_hours"].mean(),
        })
pd.DataFrame(group_rows).to_csv(OUTPUTS / "part_a_delay_by_group.csv", index=False, float_format="%.4f")

modeling.select_dtypes(include="number").corr().to_csv(OUTPUTS / "part_a_correlations.csv", index=True, index_label="column", float_format="%.4f")
