import pandas as pd
import numpy as np
import os
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric
from tqdm import tqdm
import matplotlib.pyplot as plt
import corner

def convert_to_cartesian_w_errors(star_data, num_samples=50000):
    """
    Performs Monte Carlo error propagation to transform observational data 
    (ICRS frame) into the Galactocentric Cartesian frame.

    Parameters:
        star_data (pd.Series): A row of data for one star from the gaia_data.csv dataset.
        num_samples (int): The number of samples to draw for the Monte Carlo simulation.

    Returns:
        tuple: A tuple containing:
            - mean_values (np.ndarray): Mean of the 6D Cartesian coordinates (x,y,z,u,v,w).
            - std_devs (np.ndarray): Standard deviation of the 6D coordinates.
            - covariance_matrix_6d (np.ndarray): The full 6x6 covariance matrix.
            - figure (matplotlib.figure.Figure): The corner plot figure visualizing the distributions.
    """
    # Gaia errors for RA/Dec are in mas. Convert to degrees.
    ra_err_deg = star_data['ra_error'] / 3.6e6
    dec_err_deg = star_data['dec_error'] / 3.6e6

    # Draw random samples from a normal distribution for each observational parameter
    ra_samples = np.random.normal(star_data['ra'], ra_err_deg, num_samples) * u.deg
    dec_samples = np.random.normal(star_data['dec'], dec_err_deg, num_samples) * u.deg
    distance_samples = np.random.normal(star_data['dhelio_val'], star_data['dhelio_err'], num_samples) * u.kpc
    pmra_samples = np.random.normal(star_data['pmra_val'], star_data['pmra_err'], num_samples) * u.mas / u.yr
    pmdec_samples = np.random.normal(star_data['pmdec_val'], star_data['pmdec_err'], num_samples) * u.mas / u.yr
    radial_vel_samples = np.random.normal(star_data['vhelio_val'], star_data['vhelio_err'], num_samples) * u.km / u.s

    # Create astropy SkyCoord object with all the sampled data in the ICRS frame
    coords_icrs = SkyCoord(
        ra=ra_samples,
        dec=dec_samples,
        distance=distance_samples,
        pm_ra_cosdec=pmra_samples,
        pm_dec=pmdec_samples,
        radial_velocity=radial_vel_samples,
        frame='icrs'
    )

    # Transform coordinates to the Galactocentric frame
    coords_galcen = coords_icrs.transform_to(Galactocentric)

    # Stack all 6D phase-space coordinates into a single array
    all_samples_6d = np.vstack([
        coords_galcen.x.to_value(u.kpc),
        coords_galcen.y.to_value(u.kpc),
        coords_galcen.z.to_value(u.kpc),
        coords_galcen.v_x.to_value(u.km/u.s),
        coords_galcen.v_y.to_value(u.km/u.s),
        coords_galcen.v_z.to_value(u.km/u.s)
    ])
    
    # Calculate statistics from the distribution of samples
    mean_values = np.mean(all_samples_6d, axis=1)
    std_devs = np.std(all_samples_6d, axis=1)
    covariance_matrix_6d = np.cov(all_samples_6d)

    # Generate corner plot for visualization
    labels = ["X (kpc)", "Y (kpc)", "Z (kpc)", "U (km/s)", "V (km/s)", "W (km/s)"]
    figure = corner.corner(
        all_samples_6d.T,  # Transpose is required for corner's expected format
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        verbose=False
    )

    return mean_values, std_devs, covariance_matrix_6d, figure

def main():
    """
    This script performs the second step of the data processing pipeline.
    It loads the cleaned observational data and uses a Monte Carlo simulation
    to convert it to a 6D Galactocentric Cartesian frame.
    It saves the final mean states, standard deviations, covariance matrices,
    and diagnostic corner plots.
    """
    print("--- Running make_cartesian_data.py ---")
    
    # --- 1. Set up file paths ---
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        root = os.path.abspath(os.path.join(script_dir, os.pardir))
    except NameError:
        script_dir = os.getcwd()
        root = os.path.abspath(os.path.join(script_dir, os.pardir))

    input_path = os.path.join(root, 'data', 'processed', 'gaia_data.csv')
    cartesian_output_path = os.path.join(root, 'data', 'processed', '6d_cartesian_data.csv')
    covariance_output_path = os.path.join(root, 'data', 'processed', '6d_cartesian_covariance.csv')
    corner_plots_dir = os.path.join(root, 'data', 'processed', 'corner_plots')

    os.makedirs(corner_plots_dir, exist_ok=True)

    print(f"\n[1/4] Loading Gaia data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: Processed Gaia data file not found at '{input_path}'.")
        print("Please run 'get_gaia_data.py' first.")
        return
        
    df = pd.read_csv(input_path)
    print("Gaia data loaded successfully.")

    # --- 2. Process each star ---
    print("\n[2/4] Processing each star with Monte Carlo simulation...")
    cartesian_results = []
    covariance_results = []
    
    for index, star in tqdm(df.iterrows(), total=df.shape[0], desc="Converting Stars"):
        hvs_id = int(star['HVS'])
        
        means, stds, cov_6d, fig = convert_to_cartesian_w_errors(star)
        
        var_names = ['x', 'y', 'z', 'u', 'v', 'w']
        
        # Store mean values and standard deviations
        cartesian_data = {'HVS': hvs_id, 'source_id': int(star['source_id'])}
        for i, name in enumerate(var_names):
            cartesian_data[name] = means[i]
            cartesian_data[f'{name}_err'] = stds[i]
        cartesian_results.append(cartesian_data)

        # Store the upper triangle of the covariance matrix
        covariance_data = {'HVS': hvs_id, 'source_id': int(star['source_id'])}
        for i in range(6):
            for j in range(i, 6):
                header = f'cov_{var_names[i]}{var_names[j]}'
                covariance_data[header] = cov_6d[i, j]
        covariance_results.append(covariance_data)

        # --- Save corner plot figure ---
        fig.suptitle(f"HVS #{hvs_id} Corner Plot", fontsize=16, y=1.02)
        plot_filename = f"hvs_{hvs_id}_corner.png"
        fig.savefig(os.path.join(corner_plots_dir, plot_filename), dpi=150)
        plt.close(fig)

    print("All stars processed.")
    
    # --- 3. Save the processed data to CSV files ---
    print("\n[3/4] Saving processed data...")
    
    cartesian_df = pd.DataFrame(cartesian_results)
    cartesian_df.to_csv(cartesian_output_path, index=False)
    print(f"Cartesian data saved to: {cartesian_output_path}")

    covariance_df = pd.DataFrame(covariance_results)
    covariance_df.to_csv(covariance_output_path, index=False)
    print(f"Covariance matrices saved to: {covariance_output_path}")

    print(f"\n[4/4] Diagnostic corner plots saved in: {corner_plots_dir}")
    print("\n--- Script finished ---")

if __name__ == '__main__':
    main()