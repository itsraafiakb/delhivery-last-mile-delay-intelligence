# Delhivery Last-Mile Delay Intelligence Pipeline

A reproducible, offline capstone pipeline built from the supplied synthetic shipment generator. It prepares shipment features once, estimates delivery time, flags likely promise misses, assigns operating playbooks to hubs, and turns a new batch into an action list. No real Delhivery records are used.

## How to run

From the repository root in a fresh clone, create and activate a virtual environment, install the pinned packages, and run the scripts in this order:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`; the remaining commands are the same. `generate_data.py` is committed verbatim and writes the three input files under `data/`. `package_model.py` regenerates `models/delay_model.pkl` and the model metadata from the Part C configuration. The committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`; when installed versions differ, rerun `python package_model.py` before scoring. All scripts run offline after package installation and use fixed random seeds.

## Data findings

- Express parcels miss their promise most often: 119 of 243 shipments were delayed, a rate of **0.4897**. Economy was lowest at **0.0857** (`outputs/part_a_delay_by_group.csv`).
- The delay rate rises from **0.0400** for Tier 1 to **0.4400** for Tier 3 hubs (`outputs/part_a_delay_by_group.csv`).
- Distance bands show the same gradient: **0.0219** at 0–200 km, **0.1368** at 201–400 km, **0.4364** at 401–600 km, and **0.8523** above 600 km (`outputs/part_a_delay_by_group.csv`).
- The three largest absolute correlations with delivery hours are `promised_hours` (**0.8372**), `distance_km` (**0.4013**) and `n_stops` (**0.3247**) (`outputs/part_a_correlations.csv`).
- Declared value is strongly right-skewed before transformation: the filled value mean and median are **INR 1,533.0542** and **INR 1,121.5000**. After the natural log, the mean and median are **7.0277** and **7.0224**, so the transformation compresses the long upper tail (`outputs/part_a_checks.csv`).

## Model results (regression and classification)

### Regression

- The mean baseline has validation R² **−0.0212**: predicting the training mean explains none of the validation variation and is slightly worse than using the validation mean as a reference (`outputs/part_b_metrics.csv`).
- Linear regression has the best validation RMSE at **4.8912 hours**, narrowly ahead of gradient boosting at **4.9165 hours**. The final test row is for linear regression, with R² **0.9493** and RMSE **4.5235 hours** (`outputs/part_b_metrics.csv`).
- A depth-1 tree has high bias: its training and validation RMSE are **11.6334** and **12.1324**. The unrestricted tree fits training data perfectly (RMSE **0.0000**) but has validation RMSE **7.7890**, showing high variance. Depth 6 is the selected tree and has the lowest validation RMSE (**6.2796**) (`outputs/part_b_tree_depth.csv`).
- The learning-rate experiment favors **0.1** (validation R² **0.9306**). At **0.01**, boosting learns too slowly within 100 estimators (R² **0.7127**); at **1.0**, its updates overshoot and validation R² falls to **0.8440** (`outputs/part_b_gb_learning_rate.csv`).
- Cross-validation slightly favors gradient boosting (mean R² **0.9352**, versus **0.9261** for linear regression), while the held-out validation split favors linear regression. I would ship linear regression for this run because its validation RMSE is lower; the difference is small and should be rechecked on later time-based data (`outputs/part_b_cv.csv`, `outputs/part_b_metrics.csv`).

### Classification

- The blind “all on time” baseline has accuracy **0.7667**, but recall and F1 are both **0.0000**: it catches none of the delayed parcels (`outputs/part_c_models.csv`).
- The five bookkeeping noise columns account for **0.2245** of the 17-column forest importance, using the rounded per-feature importances in the output. All five have `f_rank` **at least 8** and F-scores below **10**; none enters the selected top seven (`outputs/part_c_feature_scores.csv`). The strongest seven F-scores are for `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`.
- The feature-selection table peaks at **k = 7** with validation F1 **0.8333**; k = 5 is **0.6988**, k = 10 is **0.7857**, k = 12 is **0.8049**, and all 17 features give **0.7848** (`outputs/part_c_selectk.csv`).
- The untuned forest on those seven features scores validation F1 **0.8333**, above grid search (**0.7750**) and randomized search (**0.7848**). With only 180 validation parcels, this is evidence against choosing a configuration from a single small holdout alone, not proof that tuning is never useful (`outputs/part_c_tuning.csv`).
- The selected alert threshold is **0.5**. On validation it gives precision **0.7955** and recall **0.8333**; this is the highest tested threshold that keeps recall at or above **0.80** (`outputs/part_c_thresholds.csv`, `outputs/part_c_final_config.csv`).
- On test, precision is **0.6889**, recall **0.7561** and F1 **0.7209**. That is about **6.9 of every 10 flagged parcels** truly delayed and **7.6 of every 10 delayed parcels** caught. Test recall is lower than validation recall; the test contains only **41 delayed parcels**, so a few outcomes materially change the rate (`outputs/part_c_final_test.csv`).

## Operations recommendation

Use the segment and a parcel-level delay probability together when dispatching proactive work. The nine hubs in the **Escalate** segment and their historical hub delay rates are HUB13 Nagpur (**0.4600**), HUB15 Kochi (**0.3600**), HUB17 Patna (**0.4600**), HUB18 Ranchi (**0.3800**), HUB19 Guwahati (**0.6200**), HUB20 Raipur (**0.4800**), HUB21 Bhubaneswar (**0.4000**), HUB23 Siliguri (**0.4400**) and HUB24 Agartala (**0.5400**) (`outputs/hub_segments.csv`, `outputs/hub_profile.csv`).

The tier mismatches are HUB07 Kolkata (Tier 1, Watch), HUB13 Nagpur (Tier 2, Escalate), HUB15 Kochi (Tier 2, Escalate) and HUB22 Dehradun (Tier 3, Watch) (`outputs/tier_vs_segment_mismatch.csv`). Treat tier as a broad capacity label, not as a substitute for observed operating behavior or the segment playbook.

At threshold **0.5**, the test set suggests that about **6.9 of every 10 alerts** identify a delayed parcel and **7.6 of every 10 delayed parcels** are caught (`outputs/part_c_final_test.csv`). In the new batch, **8 of 48** parcels are flagged; **6** of those route to a hub-manager escalation, and the other **2** get a proactive customer call (`outputs/new_shipment_actions.csv`).

Keep **K = 3** because operations has three playbooks and the three-cluster solution already separates the hub profiles well (across/within ratio **5.3928**, cluster sizes **9-8-7**). Although the ratio rises to **6.3047** at K = 4 and **6.8866** at K = 5, those solutions contain clusters of **4** and **2** hubs, too small to support distinct playbooks (`outputs/part_d_separation.csv`).

## Production plan

- **Accuracy:** retain the regression test RMSE and the classifier precision, recall and F1 in the CSV outputs as release criteria.
- **Reliability:** `package_model.py` reloads the saved pipeline and records a probability difference of **0.0000** on the test rows (`outputs/part_d_reload_check.csv`).
- **Reproducibility:** requirements pin pandas, NumPy, scikit-learn and joblib; scripts use the specified random seeds and the README gives the execution order.
- **Performance:** `predict_delay.py` scores the incoming batch in one vectorized `predict_proba` call and writes one row per parcel, avoiding per-parcel model setup.
- **Maintainability:** each analytical part has its own script; shared constants and preprocessing live in `src/shipment_prep.py`.
- **Consistency:** training and batch scoring both use `clean_shipments` and `build_features`, and scoring reads the stored training medians and feature order.
- **Monitorability:** `outputs/new_shipment_actions.csv`, classifier thresholds and test metrics provide auditable predictions, actions and outcome measures.

Monitor these quantities each week after outcomes mature; the proposed thresholds are recorded in `outputs/production_monitoring_thresholds.csv`:

1. Delivered-parcel delay share: investigate if it exceeds **0.30** or drops below **0.15** for two consecutive weeks.
2. Share of incoming parcels flagged: investigate if it exceeds **0.35** or falls below **0.10** for two consecutive weeks.
3. Recall at the deployed **0.5** threshold, measured on parcels whose outcomes are now known: review the threshold and model if rolling four-week recall falls below **0.70**.
4. Input drift: investigate if population stability index exceeds **0.20** for `distance_km` or if the missing-value share for any imputed field exceeds **0.05**.

Retrain quarterly, or earlier after a trigger is confirmed and a refreshed validation shows a sustained change. During the **15–28 September** sale window, compare delivery and delay behavior with the same sale period in prior years or another comparable promotion; do not compare it directly with ordinary weeks. Data drift would appear as sustained shifts in distance, service mix, hub mix, rain or missingness, or as a different relationship between those features and delayed outcomes.

## Limitations

- The records and outcome rule are synthetic; real carrier data may contain different causes, distributions and operational constraints.
- The test set has only **41 delayed parcels**, so precision and recall have meaningful sampling uncertainty.
- The **0.5** threshold was selected on one 180-row validation split, and the business cost of an unnecessary call versus a missed delay is not explicitly priced.
- Hub segments use 50 generated shipments per hub and historical averages; actual hub behavior can change, and a hub-level segment cannot explain an individual parcel on its own.
- Delivery-hours predictions are point estimates without prediction intervals, and the model does not include external traffic, address quality or live hub load.
