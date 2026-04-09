import pandas as pd

df = pd.read_csv('cleaned_data_final.csv')

# Compute mean tot_prcp_mm per zip+month using non-null rows (2018-2021)
zip_month_mean = (
    df.dropna(subset=['tot_prcp_mm'])
      .groupby(['zip', 'month'])['tot_prcp_mm']
      .mean()
      .rename('prcp_mean')
)

df = df.join(zip_month_mean, on=['zip', 'month'])

mask = df['tot_prcp_mm'].isna()
df.loc[mask, 'tot_prcp_mm'] = df.loc[mask, 'prcp_mean']
df = df.drop(columns=['prcp_mean'])

still_null = df['tot_prcp_mm'].isna().sum()
print(f"Filled {mask.sum():,} missing tot_prcp_mm values")
print(f"Still null after fill: {still_null:,}")

df.to_csv('cleaned_data_final.csv', index=False)
print(f"Saved → cleaned_data_final.csv")
