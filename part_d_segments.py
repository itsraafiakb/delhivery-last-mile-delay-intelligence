"""Profile hubs, calculate K-Means separation evidence, and assign playbooks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
table = pd.read_csv(ROOT / "data" / "modelling_table.csv")
hub_profile = table.groupby("hub_id", as_index=False, sort=True).agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
)
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
hub_profile.to_csv(OUTPUTS / "hub_profile.csv", index=False, float_format="%.4f")

profile_columns = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(hub_profile[profile_columns])
separation_rows = []
labels_by_k = {}
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    labels_by_k[k] = labels
    all_distances = pdist(scaled, metric="euclidean")
    across = []
    within = []
    pair_index = 0
    for i in range(len(labels) - 1):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else across).append(all_distances[pair_index])
            pair_index += 1
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    separation_rows.append({
        "K": k,
        "across_over_within": float(np.mean(across) / np.mean(within)),
        "cluster_sizes": "-".join(map(str, sizes)),
    })
pd.DataFrame(separation_rows).to_csv(OUTPUTS / "part_d_separation.csv", index=False, float_format="%.4f")

labels = labels_by_k[3]
cluster_delay = {
    int(cluster): hub_profile.loc[labels == cluster, "delay_rate"].mean()
    for cluster in np.unique(labels)
}
ordered_clusters = sorted(cluster_delay, key=lambda cluster: cluster_delay[cluster])
names = {ordered_clusters[0]: "Standard", ordered_clusters[1]: "Watch", ordered_clusters[2]: "Escalate"}
hub_segments = hub_profile[["hub_id"]].copy()
hub_segments["segment"] = [names[int(label)] for label in labels]
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
hub_segments = hub_segments.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one")
hub_segments.to_csv(OUTPUTS / "hub_segments.csv", index=False)

segment_rows = []
for segment in ["Standard", "Watch", "Escalate"]:
    ids = hub_segments.loc[hub_segments["segment"] == segment, "hub_id"]
    hub_values = hub_profile[hub_profile["hub_id"].isin(ids)]
    segment_rows.append({
        "segment": segment,
        "hubs": len(hub_values),
        "mean_delivery_hours": hub_values["mean_delivery_hours"].mean(),
        "delay_rate": hub_values["delay_rate"].mean(),
        "mean_distance_km": hub_values["mean_distance_km"].mean(),
        "mean_n_stops": hub_values["mean_n_stops"].mean(),
    })
pd.DataFrame(segment_rows).to_csv(OUTPUTS / "segment_profile.csv", index=False, float_format="%.4f")

expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatches = hub_segments[hub_segments.apply(lambda row: row["segment"] != expected_segment[int(row["hub_tier"])], axis=1)]
mismatches.to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)
