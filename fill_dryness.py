import pandas as pd

df = pd.read_csv('cleaned_data_final.csv')

mask = df['dryness_index'].isna()
df.loc[mask, 'dryness_index'] = (
    df.loc[mask, 'avg_tmax_c'] / (df.loc[mask, 'tot_prcp_mm'] + 1)
)

print(f"Filled {mask.sum():,} missing dryness_index values")
print(f"Still null: {df['dryness_index'].isna().sum():,}")

df.to_csv('cleaned_data_final.csv', index=False)
print("Saved → cleaned_data_final.csv")
