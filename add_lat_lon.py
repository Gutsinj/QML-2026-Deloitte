import pandas as pd
import pgeocode

df = pd.read_csv('cleaned_data_final.csv')

nomi = pgeocode.Nominatim('us')

unique_zips = df['zip'].unique()
print(f"Looking up {len(unique_zips):,} unique zip codes...")

zip_str = pd.Series(unique_zips).astype(str).str.zfill(5)
results = nomi.query_postal_code(zip_str.tolist())

zip_coords = pd.DataFrame({
    'zip': unique_zips,
    'lat_new': results['latitude'].values,
    'lon_new': results['longitude'].values,
})

df = df.merge(zip_coords, on='zip', how='left')

df['latitude']  = df['lat_new']
df['longitude'] = df['lon_new']
df = df.drop(columns=['lat_new', 'lon_new'])

missing = df['latitude'].isna().sum()
print(f"Zip codes with no geocode result: {missing:,} rows")

df.to_csv('cleaned_data_final.csv', index=False)
print(f"Saved → cleaned_data_final.csv  (shape: {df.shape})")
