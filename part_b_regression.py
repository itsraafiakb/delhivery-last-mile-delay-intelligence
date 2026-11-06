"""Leakage-aware delivery-hours regression comparison."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {"metric": "train_scaled_max_abs_mean", "value": float(np.abs(train_scaled.mean(axis=0)).max())},
    {"metric": "test_scaled_max_abs_mean", "value": float(np.abs(test_scaled.mean(axis=0)).max())},
]).to_csv(OUTPUTS / "part_b_checks.csv", index=False, float_format="%.4f")

def score(y_true, prediction):
    mse = mean_squared_error(y_true, prediction)
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, prediction),
    }


depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": score(y_train, model.predict(X_train))["RMSE"],
        "val_rmse": score(y_val, model.predict(X_val))["RMSE"],
    })
pd.DataFrame(depth_rows).to_csv(OUTPUTS / "part_b_tree_depth.csv", index=False, float_format="%.4f")
best_depth_row = min(depth_rows, key=lambda row: (row["val_rmse"], float("inf") if row["max_depth"] == "unlimited" else row["max_depth"]))
best_depth = None if best_depth_row["max_depth"] == "unlimited" else int(best_depth_row["max_depth"])

candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
metric_rows = []
fitted = {}
val_results = {}
baseline_prediction = np.full(len(y_val), y_train.mean())
metric_rows.append({"model": "mean_baseline", "split": "val", **score(y_val, baseline_prediction)})
for name, model in candidates.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    val_score = score(y_val, model.predict(X_val))
    val_results[name] = val_score
    metric_rows.append({"model": name, "split": "val", **val_score})

gb_rows = []
for learning_rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=learning_rate, random_state=42).fit(X_train, y_train)
    prediction = model.predict(X_val)
    gb_rows.append({"learning_rate": learning_rate, "val_rmse": score(y_val, prediction)["RMSE"], "val_r2": r2_score(y_val, prediction)})
pd.DataFrame(gb_rows).to_csv(OUTPUTS / "part_b_gb_learning_rate.csv", index=False, float_format="%.4f")

ranked = sorted(candidates, key=lambda name: val_results[name]["RMSE"])
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranked[:2]:
    scores = cross_val_score(candidates[name], X_train_val, y_train_val, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).to_csv(OUTPUTS / "part_b_cv.csv", index=False, float_format="%.4f")

winner = ranked[0]
test_score = score(y_test, fitted[winner].predict(X_test))
metric_rows.append({"model": winner, "split": "test", **test_score})
pd.DataFrame(metric_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).to_csv(
    OUTPUTS / "part_b_metrics.csv", index=False, float_format="%.4f",
)
