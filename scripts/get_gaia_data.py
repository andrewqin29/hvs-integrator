import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from astroquery.gaia import Gaia

# set up paths for input and output
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # goes one level up from scripts --> root
input_path = os.path.join(project_root, 'data', 'raw', 'source_data.csv')
output_path = os.path.join(project_root, 'data', 'processed', 'gaia_data.csv')

# read df and get source ids 
df = pd.read_csv(input_path, delimiter=',')
source_ids = tuple(df['source_id'].tolist())

# clean columns, converting to numeric and splitting into val and error by ± delimeter
cols_to_clean = ['vhelio', 'pmra', 'pmdec', 'dhelio']
for col in cols_to_clean:
    split_data = df[col].str.split(' ± ', expand=True)
    df[f"{col}_val"] = pd.to_numeric(split_data[0])
    df[f"{col}_err"] = pd.to_numeric(split_data[1])

# drop cleaned and unnecessary columns
cols_to_drop = ['vhelio', 'pmra', 'pmdec', 'dhelio', 'pMW', 'pLMC', 'log_z_MW']
df.drop(columns=cols_to_drop, inplace=True)

# query gaia dr3 databse for ra dec values via adql
query = f"""
SELECT
    source_id, ra, dec, ra_error, dec_error, parallax, parallax_error
FROM
    gaiadr3.gaia_source
WHERE
    source_id IN {source_ids}
"""
print("Launching job to get results from Gaia database...")
job = Gaia.launch_job_async(query)
gaia_results = job.get_results()

print("Results found. Merging with original database...")
gaia_df = gaia_results.to_pandas()
df = pd.merge(df, gaia_df, on='source_id')
final_column_order = [
    'HVS',
    'source_id',
    'ra',
    'ra_error',
    'dec',
    'dec_error',
    'vhelio_val',
    'vhelio_err',
    'pmra_val',
    'pmra_err',
    'pmdec_val',
    'pmdec_err',
    'dhelio_val',
    'dhelio_err',
    'parallax',
    'parallax_error'
]
df = df[final_column_order]
print(df)

# output to final csv
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df.to_csv(output_path, index=False)

print(f"Data saved to '{output_path}'")