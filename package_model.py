"""Fit and package the final Part C classifier and its preprocessing metadata."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, IMPUTE_COLUMNS, build_features, clean_shipments


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
config = pd.read_csv(OUTPUTS / "part_c_final_config.csv").iloc[0]
X, y = table[FEATURES], table["is_delayed"].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
)
depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(
        n_estimators=int(config["n_estimators"]), max_depth=depth,
        min_samples_leaf=int(config["min_samples_leaf"]), random_state=42,
    )),
]).fit(X_train, y_train)

joblib.dump(model, MODELS / "delay_model.pkl")
pd.DataFrame({"threshold": [float(config["threshold"])]}).to_csv(MODELS / "delay_threshold.csv", index=False, float_format="%.4f")
pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)
raw = pd.read_csv(ROOT / "data" / "raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame([{"column": column, "median": medians[column]} for column in IMPUTE_COLUMNS]).to_csv(
    MODELS / "impute_values.csv", index=False, float_format="%.4f",
)

reloaded = joblib.load(MODELS / "delay_model.pkl")
in_memory_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded.predict_proba(X_test)[:, 1]
pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {"metric": "reload_max_abs_probability_diff", "value": float(np.max(np.abs(in_memory_probabilities - reloaded_probabilities)))},
    {"metric": "threshold", "value": float(config["threshold"])},
]).to_csv(OUTPUTS / "part_d_reload_check.csv", index=False, float_format="%.4f")
