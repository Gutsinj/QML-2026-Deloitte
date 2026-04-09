import pandas as pd
import numpy as np
from itertools import product

# Load data
df = pd.read_csv('cleaned_data_sorted.csv')

# ── 1. Filter to 2018-2022 ────────────────────────────────────────────────────
df = df[df['year'].between(2018, 2022)].copy()
print(f"After year filter: {len(df):,} rows")

# ── 1b. Deduplicate zip+year+month ────────────────────────────────────────────
# Some zips have multiple rows per month (different fire locations, or one row
# with weather data and one with fire coords).  Resolve by:
#   - has_fire = 1 if ANY row for that zip+month has a fire
#   - weather columns: use the row that has valid data (non-NaN avg_tmax_c)
#   - lat/lon: prefer fire-row values when available
KEY = ['zip', 'year', 'month']
WEATHER_COLS = ['avg_tmax_c', 'avg_tmin_c', 'temp_mean', 'temp_range',
                'tot_prcp_mm', 'dryness_index']

# Count fires per zip+year+month before deduplication
fire_counts = (
    df[df['has_fire'] == 1]
    .groupby(KEY)
    .size()
    .rename('num_fires')
)

before = len(df)

# Sort so rows with valid avg_tmax_c come first, fire rows with lat come next
df['_has_weather'] = df['avg_tmax_c'].notna().astype(int)
df['_is_fire']     = (df['has_fire'] == 1).astype(int)
df['_has_lat']     = df['latitude'].notna().astype(int)
df = df.sort_values(
    ['zip', 'year', 'month', '_has_weather', '_is_fire', '_has_lat'],
    ascending=[True, True, True, False, False, False]
)
df = df.drop(columns=['_has_weather', '_is_fire', '_has_lat'])

# Per group: take first weather values (from best-sorted row), OR any valid value
# Step A – fire flag: 1 if any row in group fired
fire_flag = df.groupby(KEY)['has_fire'].transform('max').astype(int)
df['has_fire'] = fire_flag

# Step B – weather: forward-fill from the best row (already sorted to front)
# Keep first non-null value per group for each weather column
for col in WEATHER_COLS:
    first_valid = df.groupby(KEY)[col].transform('first')
    df[col] = df[col].where(df[col].notna(), first_valid)

# Step C – lat/lon: take from fire rows with valid lat (sorted to front for has_fire groups)
for col in ['latitude', 'longitude']:
    first_valid = df.groupby(KEY)[col].transform('first')
    df[col] = df[col].where(df[col].notna(), first_valid)

# Drop duplicates – keep first row per group (which has the best data after sorting)
df = df.drop_duplicates(subset=KEY, keep='first').reset_index(drop=True)
print(f"After dedup: {len(df):,} rows  (removed {before - len(df):,} duplicates)")

# Attach fire counts (0 for rows with no fires)
df = df.join(fire_counts, on=KEY)
df['num_fires'] = df['num_fires'].fillna(0).astype(int)

# ── 2. Build the full expected index (every zip × year × month) ───────────────
zips   = df['zip'].unique()
years  = range(2018, 2023)
months = range(1, 13)

full_index = pd.DataFrame(
    list(product(zips, years, months)),
    columns=['zip', 'year', 'month']
)
full_index['year_month'] = (
    full_index['year'].astype(str) + '-' +
    full_index['month'].astype(str).str.zfill(2)
)

# Merge to find missing combos
df = full_index.merge(df, on=['zip', 'year', 'month', 'year_month'], how='left')

# Newly created rows: set has_fire = 0 and num_fires = 0 (no fire assumed)
df['has_fire']  = df['has_fire'].fillna(0).astype(int)
df['num_fires'] = df['num_fires'].fillna(0).astype(int)

print(f"After adding missing combos: {len(df):,} rows  "
      f"(expected {len(zips) * 5 * 12:,})")

# ── 3. Fill avg_tmax_c / avg_tmin_c with zip+month historical mean ────────────
# Use only the 2018-2022 data itself (all years available for that zip+month)
for col in ['avg_tmax_c', 'avg_tmin_c']:
    zip_month_mean = (
        df.dropna(subset=[col])
          .groupby(['zip', 'month'])[col]
          .mean()
          .rename(f'{col}_mean')
    )
    df = df.join(zip_month_mean, on=['zip', 'month'])
    mask = df[col].isna()
    df.loc[mask, col] = df.loc[mask, f'{col}_mean']
    df.drop(columns=[f'{col}_mean'], inplace=True)
    filled = mask.sum()
    print(f"Filled {filled:,} missing {col} values using zip+month average")

# ── 4. Re-derive temp_mean and temp_range where they are missing ───────────────
missing_mean  = df['temp_mean'].isna()
missing_range = df['temp_range'].isna()
df.loc[missing_mean,  'temp_mean']  = (df.loc[missing_mean,  'avg_tmax_c'] + df.loc[missing_mean,  'avg_tmin_c']) / 2
df.loc[missing_range, 'temp_range'] = (df.loc[missing_range, 'avg_tmax_c'] - df.loc[missing_range, 'avg_tmin_c'])
print(f"Re-derived {missing_mean.sum():,} temp_mean and {missing_range.sum():,} temp_range values")

# ── 5. Sort and save ──────────────────────────────────────────────────────────
df = df.sort_values(['zip', 'year', 'month']).reset_index(drop=True)

out_path = 'cleaned_data_final.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(f"Final shape: {df.shape}")
print(f"Remaining nulls:\n{df.isna().sum()[df.isna().sum() > 0]}")
