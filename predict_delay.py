"""Score the new shipment batch and assign operational actions."""

from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments


ROOT = Path(__file__).resolve().parent
raw = pd.read_csv(ROOT / "data" / "new_shipments.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
medians_table = pd.read_csv(ROOT / "models" / "impute_values.csv")
medians = dict(zip(medians_table["column"], medians_table["median"].astype(float)))
threshold = float(pd.read_csv(ROOT / "models" / "delay_threshold.csv").iloc[0, 0])
feature_order = pd.read_csv(ROOT / "models" / "feature_order.csv")["feature"].tolist()
segments = pd.read_csv(ROOT / "outputs" / "hub_segments.csv")[["hub_id", "segment"]]
model = joblib.load(ROOT / "models" / "delay_model.pkl")

clean, _ = clean_shipments(raw, medians=medians)
featured = build_features(clean, hubs)
probability = model.predict_proba(featured[feature_order])[:, 1]
result = clean[["shipment_id", "hub_id"]].copy()
result["delay_probability"] = pd.Series(probability, index=result.index).round(4)
result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
result = result.merge(segments, on="hub_id", how="left", validate="many_to_one")
result["action"] = "No action"
flagged = result["flagged"].eq(1)
result.loc[flagged & result["segment"].eq("Escalate"), "action"] = "Escalate to hub manager"
result.loc[flagged & ~result["segment"].eq("Escalate"), "action"] = "Proactive customer call"
result = result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
result.to_csv(ROOT / "outputs" / "new_shipment_actions.csv", index=False, float_format="%.4f")
