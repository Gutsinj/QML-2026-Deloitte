"""
Task 2: Insurance Premium Prediction — Ridge Regression
========================================================
Predicts 2021 ZIP-level insurance premium rates using Ridge regression
trained on 2019-2020 data (requiring one lag year from 2018-2019).

Target : log(Earned Premium / Earned Exposure) — premium rate per policy
Train  : 2019-2020 (lag features from 2018-2019)
Test   : 2021      (lag features from 2020)

Outputs:
  task2_predictions_2021.csv   — predicted vs actual per ZIP
  task2_metrics.txt            — accuracy metrics
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("original data")
INS_CSV  = DATA_DIR / "abfa2rbci2UF6CTj_cal_insurance_fire_census_weather.csv"
QML_CSV  = Path("qml_annual_risk_2023.csv")
OUT_CSV  = Path("task2_predictions_2021.csv")
OUT_TXT  = Path("task2_metrics.txt")

# ── 1. Load & clean ────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(INS_CSV)
df = df[(df["Earned Exposure"] > 0) & (df["Earned Premium"] > 0)].copy()
df["ZIP"]  = df["ZIP"].astype(int)
df["Year"] = df["Year"].astype(int)
df["premium_rate"] = df["Earned Premium"] / df["Earned Exposure"]
df["log_rate"]     = np.log(df["premium_rate"])

# ── 2. Aggregate to one row per ZIP × Year ─────────────────────────────────────
panel = (
    df.groupby(["ZIP", "Year"])
    .agg(
        log_rate        = ("log_rate",                    "first"),
        premium_rate    = ("premium_rate",                "first"),
        fire_risk       = ("Avg Fire Risk Score",         "mean"),
        cov_a           = ("Cov A Amount Weighted Avg",   "first"),
        cov_c           = ("Cov C Amount Weighted Avg",   "first"),
        exposure        = ("Earned Exposure",             "first"),
        income          = ("median_income",               "first"),
        house_val       = ("housing_value",               "first"),
        ppc             = ("Avg PPC",                     "mean"),
        cat_losses      = ("CAT Cov A Fire -  Incurred Losses", "sum"),
    )
    .reset_index()
    .sort_values(["ZIP", "Year"])
)

# ── 3. Merge QML fire risk scores ──────────────────────────────────────────────
if QML_CSV.exists():
    qml = pd.read_csv(QML_CSV).rename(columns={"zip":"ZIP","qml3_pred":"qml_risk"})
    qml["ZIP"] = qml["ZIP"].astype(int)
    panel = panel.merge(qml[["ZIP","qml_risk"]], on="ZIP", how="left")
else:
    panel["qml_risk"] = np.nan

# ── 4. Lag features (t-1) ──────────────────────────────────────────────────────
for col in ["log_rate", "fire_risk", "exposure", "cat_losses"]:
    panel[f"lag_{col}"] = panel.groupby("ZIP")[col].shift(1)

# yoy_change must be LAGGED — current year's change would leak the target
panel["yoy_change_raw"] = panel["log_rate"] - panel["lag_log_rate"]
panel["lag_yoy_change"] = panel.groupby("ZIP")["yoy_change_raw"].shift(1)

# Drop rows missing any lag (= year 2018, first year in the panel)
panel = panel.dropna(subset=["lag_log_rate"])

# Fill remaining NAs with column median
FEATURES = [
    "lag_log_rate",     # prior-year premium rate  (most predictive)
    "lag_yoy_change",   # prior-year momentum (lagged to avoid leakage)
    "fire_risk",        # current fire risk score
    "lag_fire_risk",    # prior-year fire risk
    "qml_risk",         # QML model risk score
    "cov_a",            # avg dwelling coverage amount
    "cov_c",            # avg contents coverage amount
    "exposure",         # policy count proxy
    "lag_exposure",
    "lag_cat_losses",   # prior-year catastrophe losses
    "income",           # median income
    "house_val",        # median home value
    "ppc",              # protection class code
    "Year",             # time trend
]
FEATURES = [f for f in FEATURES if f in panel.columns]

for f in FEATURES:
    panel[f] = panel[f].fillna(panel[f].median())

# ── 5. Train / test split ──────────────────────────────────────────────────────
train = panel[panel["Year"].isin([2019, 2020])]
test  = panel[panel["Year"] == 2021]

X_train = train[FEATURES].values
y_train = train["log_rate"].values
X_test  = test[FEATURES].values
y_test  = test["log_rate"].values

print(f"  Train rows : {len(train):,}  |  Test rows: {len(test):,}")
print(f"  Features   : {len(FEATURES)}")

# ── 6. Scale + fit Ridge (alpha selected by cross-validation) ─────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# RidgeCV searches over a log-spaced alpha grid on training data
alphas = np.logspace(-2, 4, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
ridge_cv.fit(X_train_s, y_train)
best_alpha = ridge_cv.alpha_

ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_s, y_train)

print(f"\n  Ridge CV best alpha = {best_alpha:.4f}")

# ── 7. Evaluate ────────────────────────────────────────────────────────────────
pred_log  = ridge.predict(X_test_s)
pred_rate = np.exp(pred_log)
true_rate = np.exp(y_test)

mae       = mean_absolute_error(true_rate, pred_rate)
rmse      = np.sqrt(mean_squared_error(true_rate, pred_rate))
r2        = r2_score(y_test, pred_log)
mape      = np.median(np.abs(pred_rate - true_rate) / true_rate) * 100
within_10 = np.mean(np.abs(pred_rate - true_rate) / true_rate < 0.10) * 100
within_20 = np.mean(np.abs(pred_rate - true_rate) / true_rate < 0.20) * 100

print(f"\n  ── 2021 Test Set Results ──────────────────────────────")
print(f"  MAE              = ${mae:>10,.2f}  per exposure unit")
print(f"  RMSE             = ${rmse:>10,.2f}  per exposure unit")
print(f"  R² (log scale)   = {r2:>10.4f}")
print(f"  Median APE       = {mape:>10.2f}%")
print(f"  Within 10%       = {within_10:>10.1f}%")
print(f"  Within 20%       = {within_20:>10.1f}%")

# Feature coefficients
coef_df = (
    pd.Series(ridge.coef_, index=FEATURES)
    .reindex(pd.Series(np.abs(ridge.coef_), index=FEATURES)
             .sort_values(ascending=False).index)
)
print(f"\n  Ridge coefficients (sorted by |coef|):")
for feat, coef in coef_df.items():
    print(f"    {feat:<25} {coef:+.4f}")

# Fire risk sensitivity
fire_idx = FEATURES.index("fire_risk")
base = X_test_s.mean(axis=0).copy()
sensitivity = {}
for pct in [10, 50, 90]:
    raw_val   = np.percentile(X_test[:, fire_idx], pct)
    scaled    = (raw_val - scaler.mean_[fire_idx]) / scaler.scale_[fire_idx]
    row       = base.copy(); row[fire_idx] = scaled
    pred      = np.exp(ridge.predict(row.reshape(1,-1))[0])
    sensitivity[pct] = (raw_val, pred)

median_pred = sensitivity[50][1]
print(f"\n  Fire risk premium sensitivity (all other features at test mean):")
for pct, (rv, pr) in sensitivity.items():
    delta = (pr - median_pred) / median_pred * 100
    marker = "  (baseline)" if pct == 50 else f"  ({delta:+.1f}% vs median)"
    print(f"    {pct}th pct  risk={rv:.3f}  ->  ${pr:,.0f}/exposure{marker}")

# ── 8. Top predicted ZIPs ─────────────────────────────────────────────────────
out = test[["ZIP","premium_rate","fire_risk"]].copy().reset_index(drop=True)
out = out.rename(columns={"premium_rate":"actual_rate_2021",
                           "fire_risk":"fire_risk_score"})
out["predicted_rate_2021"] = pred_rate
out["pct_error"]           = ((pred_rate - true_rate) / true_rate * 100).round(2)
if "qml_risk" in test.columns:
    out["qml_risk_score"] = test["qml_risk"].values

out_sorted = out.sort_values("predicted_rate_2021", ascending=False).reset_index(drop=True)

print(f"\n  Top 10 ZIPs by predicted 2021 premium rate:")
print(out_sorted.head(10)[["ZIP","predicted_rate_2021","actual_rate_2021",
                             "pct_error","fire_risk_score"]].to_string(index=False))

# Aggregate statewide trend
print(f"\n  Statewide median rate:")
for yr in [2018,2019,2020]:
    m = panel[panel["Year"]==yr]["premium_rate"].median()
    print(f"    {yr}: ${m:,.0f}")
print(f"    2021 actual  median: ${np.median(true_rate):,.0f}")
print(f"    2021 predict median: ${np.median(pred_rate):,.0f}")

# ── 9. Save outputs ────────────────────────────────────────────────────────────
out_sorted.to_csv(OUT_CSV, index=False)
print(f"\n  Saved {len(out_sorted):,} rows → {OUT_CSV}")

with open(OUT_TXT, "w") as f:
    f.write("Task 2 — Ridge Regression 2021 Insurance Premium Rate Prediction\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Model      : Ridge Regression (alpha={best_alpha:.4f}, 5-fold CV)\n")
    f.write(f"Target     : log(Earned Premium / Earned Exposure)\n")
    f.write(f"Train      : 2019-2020  ({len(train):,} ZIP-year rows)\n")
    f.write(f"Test       : 2021       ({len(test):,} ZIP-year rows)\n")
    f.write(f"Features   : {len(FEATURES)}\n\n")
    f.write(f"MAE              = ${mae:,.2f} per exposure unit\n")
    f.write(f"RMSE             = ${rmse:,.2f} per exposure unit\n")
    f.write(f"R2 (log scale)   = {r2:.4f}\n")
    f.write(f"Median APE       = {mape:.2f}%\n")
    f.write(f"% within 10%     = {within_10:.1f}%\n")
    f.write(f"% within 20%     = {within_20:.1f}%\n\n")
    f.write("Fire risk sensitivity:\n")
    for pct, (rv, pr) in sensitivity.items():
        delta = (pr - median_pred) / median_pred * 100
        f.write(f"  {pct}th pct  risk={rv:.3f}  predicted=${pr:,.0f}  "
                f"({delta:+.1f}% vs median)\n")
    f.write(f"\nTop 5 ZIPs by predicted rate:\n")
    f.write(out_sorted.head(5)[["ZIP","predicted_rate_2021",
                                 "actual_rate_2021","pct_error"]].to_string(index=False))

print(f"  Saved metrics → {OUT_TXT}")
