"""Train, tune, and threshold the delay classifier."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp,
)


def metrics(y_true, prediction):
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
    }


val_model_rows = []
val_model_rows.append({"model": "blind_all_on_time", **metrics(y_val, np.zeros(len(y_val), dtype=int))})

logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
val_model_rows.append({"model": "logistic_regression", **metrics(y_val, logistic.predict(X_val))})

tree_rows = []
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(X_train, y_train)
    tree_rows.append((depth, f1_score(y_val, tree.predict(X_val), zero_division=0), tree))
tree_depth, _, best_tree = max(tree_rows, key=lambda item: (item[1], -item[0]))
val_model_rows.append({"model": f"decision_tree_depth_{tree_depth}", **metrics(y_val, best_tree.predict(X_val))})

default_forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
val_model_rows.append({"model": "random_forest_default", **metrics(y_val, default_forest.predict(X_val))})
pd.DataFrame(val_model_rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv(
    OUTPUTS / "part_c_models.csv", index=False, float_format="%.4f",
)

selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
forest_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
scores = selector_all.scores_
order = np.argsort(-scores, kind="stable")
ranks = np.empty(len(scores), dtype=int)
ranks[order] = np.arange(1, len(scores) + 1)
pd.DataFrame({
    "feature": FEATURES,
    "f_score": scores,
    "f_rank": ranks,
    "rf_importance": forest_all.feature_importances_,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
}).to_csv(OUTPUTS / "part_c_feature_scores.csv", index=False, float_format="%.4f")

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val), zero_division=0)})
pd.DataFrame(selectk_rows).to_csv(OUTPUTS / "part_c_selectk.csv", index=False, float_format="%.4f")

selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_7 = selector.transform(X_train)
X_val_7 = selector.transform(X_val)
parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), parameter_grid, cv=5)
grid.fit(X_train_7, y_train)
randomized = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), parameter_grid, n_iter=9, cv=5, random_state=42,
)
randomized.fit(X_train_7, y_train)

def params_text(params):
    normalized = {key: (int(value) if isinstance(value, np.integer) else value) for key, value in params.items()}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


grid_val_f1 = f1_score(y_val, grid.best_estimator_.predict(X_val_7), zero_division=0)
random_val_f1 = f1_score(y_val, randomized.best_estimator_.predict(X_val_7), zero_division=0)
default_7 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, min_samples_leaf=1).fit(X_train_7, y_train)
default_val_f1 = f1_score(y_val, default_7.predict(X_val_7), zero_division=0)
tuning_rows = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
     "best_params": params_text(grid.best_params_), "best_cv_accuracy": grid.best_score_, "val_f1": grid_val_f1},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
     "best_params": params_text(randomized.best_params_), "best_cv_accuracy": randomized.best_score_, "val_f1": random_val_f1},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "n_estimators=100; max_depth=unlimited; min_samples_leaf=1", "best_cv_accuracy": np.nan, "val_f1": default_val_f1},
]
pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]).to_csv(
    OUTPUTS / "part_c_tuning.csv", index=False, float_format="%.4f",
)

priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
winner = sorted(tuning_rows, key=lambda row: (-row["val_f1"], priority[row["search"]]))[0]
if winner["search"] == "grid_search":
    chosen_params = grid.best_params_
elif winner["search"] == "randomized_search":
    chosen_params = randomized.best_params_
else:
    chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
]).fit(X_train, y_train)
val_probabilities = final_model.predict_proba(X_val)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = []
for threshold in thresholds:
    prediction = (val_probabilities >= threshold).astype(int)
    threshold_rows.append({"threshold": threshold, **metrics(y_val, prediction)})
pd.DataFrame(threshold_rows, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).to_csv(
    OUTPUTS / "part_c_thresholds.csv", index=False, float_format="%.4f",
)
eligible = [row for row in threshold_rows if row["recall"] >= 0.80]
threshold = max(eligible, key=lambda row: row["threshold"])["threshold"] if eligible else 0.1
test_prediction = (final_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
test_row = {"threshold": threshold, **metrics(y_test, test_prediction)}
pd.DataFrame([test_row]).to_csv(OUTPUTS / "part_c_final_test.csv", index=False, float_format="%.4f")
pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": int(chosen_params["n_estimators"]),
    "max_depth": "unlimited" if chosen_params["max_depth"] is None else int(chosen_params["max_depth"]),
    "min_samples_leaf": int(chosen_params["min_samples_leaf"]),
    "threshold": threshold,
    "chosen_from": winner["search"],
}]).to_csv(OUTPUTS / "part_c_final_config.csv", index=False, float_format="%.4f")
