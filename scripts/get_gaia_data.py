import pandas as pd
import numpy as np
import os
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, Galactic

def main():
    """
    This script performs the first step of the data processing pipeline.
    It takes the raw source data of hypervelocity stars, cleans the columns,
    and queries the Gaia DR3 database to get precise astrometric data.
    The final cleaned and merged dataset is saved to the processed data folder.
    """
    print("--- Running get_gaia_data.py ---")

    # --- 1. Set up file paths based on the project structure ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    root = os.path.abspath(os.path.join(script_dir, os.pardir))

    input_path = os.path.join(root, 'data', 'raw', 'source_data.csv')
    output_path = os.path.join(root, 'data', 'processed', 'gaia_data.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[1/5] Loading raw data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Raw data file not found at '{input_path}'.")
        print("Please ensure 'source_data.csv' is in the 'data/raw' directory.")
        return

    df = pd.read_csv(input_path, delimiter=',')
    print("Raw data loaded successfully.")
    
    # --- 2. Clean columns with 'value ± error' format ---
    print("\n[2/5] Cleaning data columns...")
    cols_to_clean = ['vhelio', 'pmra', 'pmdec', 'dhelio']
    for col in cols_to_clean:
        if col in df.columns:
            split_data = df[col].astype(str).str.split(' ± ', expand=True)
            df[f"{col}_val"] = pd.to_numeric(split_data[0])
            df[f"{col}_err"] = pd.to_numeric(split_data[1])
    
    # Drop original combined columns and other unneeded columns
    cols_to_drop = ['vhelio', 'pmra', 'pmdec', 'dhelio', 'pMW', 'pLMC', 'log_z_MW']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print("Columns cleaned and separated into value/error pairs.")

    # --- 3. Query Gaia DR3 database for astrometric data ---
    print("\n[3/5] Querying Gaia DR3 database...")
    source_ids = df['source_id'].tolist()

    # ADQL query from Gaia DR3
    query = f"""
    SELECT
        source_id, ra, dec, ra_error, dec_error, parallax, parallax_error
    FROM
        gaiadr3.gaia_source
    WHERE
        source_id IN {tuple(source_ids)}
    """
    
    try:
        job = Gaia.launch_job_async(query)
        gaia_results = job.get_results()
        gaia_df = gaia_results.to_pandas()
        print("Gaia query successful.")
    except Exception as e:
        print(f"Error: Failed to query Gaia. An internet connection is required. Details: {e}")
        return

    # --- 4. Merge Gaia data with the main DataFrame ---
    print("\n[4/5] Merging Gaia data with source data...")
    df = pd.merge(df, gaia_df, on='source_id')
    
    final_column_order = [
        'HVS', 'source_id',
        'ra', 'ra_error',
        'dec', 'dec_error',
        'pmra_val', 'pmra_err',
        'pmdec_val', 'pmdec_err',
        'dhelio_val', 'dhelio_err',
        'vhelio_val', 'vhelio_err',
        'parallax', 'parallax_error'
    ]
    df = df[final_column_order]
    print("Dataframes merged successfully.")

    # --- 5. Save the final processed DataFrame ---
    df.to_csv(output_path, index=False)
    print(f"\n[5/5] Final data saved to: {output_path}")
    print("\n--- Script finished ---")

if __name__ == '__main__':
    main()