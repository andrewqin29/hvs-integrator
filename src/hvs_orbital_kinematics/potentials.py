import numpy as np
from astropy import constants as const
import astropy.units as u

# express gravitational constant in terms of standard units (kpc, m_Sun, Myr)
G_KPC_MYR = const.G.to(u.kpc**3 / (u.Msun * u.Myr**2)).value

# define potential component functions which takes a 3D position vector and a dictionary of parameters
# describing the galactic component and returns the 3D acceleration vector produced by that component
# so in the end, we just sum up vectors from the 3 components: hernquist buldge, miyamoto nagai disks and NFW

def hernquist_acceleration(pos, params):
    """
    Calculates acceleration due to Hernquist Buldge potential

    Parameters:
        - pos: The 3d position vector of the HVS
        - params: A dictionary of the parameters. 'm' for mass in Suns, 'c' for scale radius (kpc)
    """
    r = np.linalg.norm(pos) # radius from center
    if r == 0:
        return np.array([0., 0., 0.]) # early check preventing divide by 0
    mass_term = G_KPC_MYR * params['m']
    denominator = r * (r + params['c'])**2

    return -mass_term / denominator * pos #directed towards galactic center

def miyamoto_nagai_acceleration(pos, params):
    pass
def nfw_acceleration(pos, params):
    pass
