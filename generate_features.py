import numpy as np
import pandas as pd

df = pd.read_csv('cleaned_data_final.csv')

# ── 1. Sort so rolling windows are computed in time order ─────────────────────
df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

# ── 2. Cyclical month encoding ────────────────────────────────────────────────
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ── 3. 3-month rolling mean of dryness_index per zip ─────────────────────────
# min_periods=1 so the first 1-2 months of each zip still get a value
df['dryness_3m_avg'] = (
    df.groupby('zip')['dryness_index']
      .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

# ── 4. Log1p target ───────────────────────────────────────────────────────────
df['target'] = np.log1p(df['num_fires'])

# ── 5. Select final feature columns + identifiers ────────────────────────────
FEATURES = [
    'month_sin',
    'month_cos',
    'year',
    'avg_tmax_c',
    'temp_range',
    'tot_prcp_mm',
    'dryness_3m_avg',
    'latitude',
    'longitude',
]

ID_COLS = ['zip', 'year_month', 'has_fire', 'num_fires', 'target']
out = df[ID_COLS + FEATURES].copy()

# ── 6. Min-max normalize all features to [0, 1] ───────────────────────────────
# Fit scaler on 2018-2022 only, then apply to all rows (including any 2023 data)
train_mask = out['year'].between(2018, 2022)  # year is in FEATURES

feature_stats = {}
for col in FEATURES:
    col_min = out.loc[train_mask, col].min()
    col_max = out.loc[train_mask, col].max()
    feature_stats[col] = (col_min, col_max)
    out[col] = (out[col] - col_min) / (col_max - col_min)
    # Clip to [0,1] in case any value falls slightly outside training range
    out[col] = out[col].clip(0, 1)

print("Feature ranges after normalization:")
print(out[FEATURES].describe().loc[['min', 'max']].T.to_string())

# ── 7. Check for any remaining nulls ─────────────────────────────────────────
nulls = out[FEATURES + ['target']].isna().sum()
if nulls.any():
    print(f"\nWarning — nulls remaining:\n{nulls[nulls > 0]}")
else:
    print("\nNo nulls in features or target.")

# ── 8. Save ───────────────────────────────────────────────────────────────────
out.to_csv('features.csv', index=False)
print(f"\nSaved → features.csv  (shape: {out.shape})")
print(f"Columns: {list(out.columns)}")

# Save scaler stats for use at inference time
stats_df = pd.DataFrame(feature_stats, index=['min', 'max']).T
stats_df.to_csv('feature_scaler_stats.csv')
print(f"Saved → feature_scaler_stats.csv")
