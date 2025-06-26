import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Galactocentric
import os
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

def convert_to_cartesian_w_errors(curr_data, num_samples=1000):
    """
    Monte Carlo error propogation for coordinate transformations into cartesian frame via Gaussian random sampling.
    """
    
    # convert errors to degrees
    ra_err_deg = curr_data['ra_error'] / 3.6e6
    dec_err_deg = curr_data['dec_error'] / 3.6e6

    # random sampling of components
    ra_samples = np.random.normal(curr_data['ra'], ra_err_deg, num_samples) * u.deg
    dec_samples = np.random.normal(curr_data['dec'], dec_err_deg, num_samples) * u.deg
    distance_samples = np.random.normal(curr_data['dhelio_val'], curr_data['dhelio_err'], num_samples) * u.kpc
    pmra_samples = np.random.normal(curr_data['pmra_val'], curr_data['pmra_err'], num_samples) * u.mas / u.yr
    pmdec_samples = np.random.normal(curr_data['pmdec_val'], curr_data['pmdec_err'], num_samples) * u.mas / u.yr
    radial_vel_samples = np.random.normal(curr_data['vhelio_val'], curr_data['vhelio_err'], num_samples) * u.km / u.s

    # create galactic frame skycoord object, convert to cartesian/galactocentric and return
    coords_galactic = SkyCoord(
        ra=ra_samples,
        dec=dec_samples,
        distance=distance_samples,
        pm_ra_cosdec=pmra_samples,
        pm_dec=pmdec_samples,
        radial_velocity=radial_vel_samples,
        frame='icrs'
    )

    coords_galcen = coords_galactic.transform_to(Galactocentric)
    cartesian_df = {
        'x': np.mean(coords_galcen.x.to_value(u.kpc)), 'x_err': np.std(coords_galcen.x.to_value(u.kpc)),
        'y': np.mean(coords_galcen.y.to_value(u.kpc)), 'y_err': np.std(coords_galcen.y.to_value(u.kpc)),
        'z': np.mean(coords_galcen.z.to_value(u.kpc)), 'z_err': np.std(coords_galcen.z.to_value(u.kpc)),
        'u': np.mean(coords_galcen.v_x.to_value(u.km/u.s)), 'u_err': np.std(coords_galcen.v_x.to_value(u.km/u.s)),
        'v': np.mean(coords_galcen.v_y.to_value(u.km/u.s)), 'v_err': np.std(coords_galcen.v_y.to_value(u.km/u.s)),
        'w': np.mean(coords_galcen.v_z.to_value(u.km/u.s)), 'w_err': np.std(coords_galcen.v_z.to_value(u.km/u.s))
    }

    return cartesian_df

df = pd.read_csv