"""Shared cleaning and feature engineering for shipment models."""

import numpy as np
import pandas as pd


PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
IMPUTE_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    """Normalize the raw export and fill the three numeric fields.

    Returns a new DataFrame and the exact median dictionary used for filling.
    """
    clean = raw.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].astype("string").str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)
    clean = clean.drop(columns=["declared_value"])
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"], errors="coerce")
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    if medians is None:
        used_medians = {column: float(clean[column].median()) for column in IMPUTE_COLUMNS}
    else:
        used_medians = {column: float(medians[column]) for column in IMPUTE_COLUMNS}
    for column in IMPUTE_COLUMNS:
        clean[column] = clean[column].fillna(used_medians[column])
    return clean, used_medians


def build_features(clean, hubs):
    """Attach hub attributes and add dispatch-time and service features."""
    features = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id", how="left", validate="many_to_one",
    )
    features["promised_hours"] = features["service_type"].map(PROMISED_HOURS).astype("int64")
    features["is_cod"] = features["payment_mode"].eq("COD").astype("int64")
    features["is_weekend"] = features["dispatch_time"].dt.dayofweek.ge(5).astype("int64")
    features["dispatch_hour"] = features["dispatch_time"].dt.hour.astype("int64")
    features["log_declared_value"] = np.log(features["declared_value_inr"])
    return features
