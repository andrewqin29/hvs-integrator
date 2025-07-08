import numpy as np
from astropy import constants as const
import astropy.units as u

# express gravitational constant in terms of standard units (kpc, m_Sun, Myr)
G_KPC_MYR = const.G.to(u.kpc**3 / (u.Msun * u.Myr**2)).value
PI = 3.1415926535

def hernquist_acceleration(pos, params):
    """
    Calculates acceleration due to Hernquist Buldge potential. Used as the acceleration function for both
    the Hernquist bulge and spherical nucleus which can be modeled as a Hernquist potential.

    Parameters:
        - pos: The 3d position vector of the HVS
        - params: A dictionary of the parameters. 'm' for mass in Suns, 'a' for scale radius (kpc)
    """
    r = np.linalg.norm(pos) # radius from center
    if r == 0:
        return np.array([0., 0., 0.]) # early check preventing divide by 0
    
    m, a = params['m'], params['a']
    mass_term = G_KPC_MYR * m
    denominator = r * (r + a)**2

    return -mass_term / denominator * pos # directed towards galactic center

def miyamoto_nagai_acceleration(pos, params):
    """
    Calculates acceleration due to a Miyamoto-Nagai Disk. Will be used for all three disks in MN3 model.

    Parameters:
        - pos: The 3d position vector of the HVS
        - params: A dictionary of the parameters. 'm' for mass in Suns, 'a' for scale length (kpc), 'h' for scale height
    """
    x, y, z = pos
    m, a, b = params['m'], params['a'], params['b']
    
    R_sq = x**2 + y**2
    sqrt_z_sq_b_sq = np.sqrt(z**2 + b**2)

    denominator = (R_sq + (a + sqrt_z_sq_b_sq)**2)**(1.5)
    
    accel_R_mag = -G_KPC_MYR * m * (a + sqrt_z_sq_b_sq) / denominator
    accel_z = -G_KPC_MYR * m * z * (a + sqrt_z_sq_b_sq) / (denominator * sqrt_z_sq_b_sq)

    R = np.sqrt(R_sq)
    if R == 0:
        accel_x, accel_y = 0., 0.
    else:
        accel_x = accel_R_mag * (x / R)
        accel_y = accel_R_mag * (y / R)
    return np.array([accel_x, accel_y, accel_z])
    

def nfw_acceleration(pos, params):
    """
    Calculates acceleration due to NFW dark matter halo potential.

    Parameters:
        - pos: The 3d position vector of the HVS
        - params: A dictionary of the parameters. 'rho' for scale density, 'a' for scale radius (kpc)
    """
    r = np.linalg.norm(pos)
    if r == 0:
        return np.array([0., 0., 0.])
    
    m, r_s = params['m'], params['r_s']
    x_norm = r / r_s
    
    mass_term = G_KPC_MYR * m
    log_term = np.log(1 + x_norm) - (x_norm / (1 + x_norm))
    denominator = r**2 * log_term

    if denominator == 0:
        return np.array([0., 0., 0.])
    return -mass_term / denominator * pos



class MWPotential:
    def __init__(self):

    
        self.components = {
            # initialize parameters for the components. see: https://gala.adrian.pw/en/latest/_modules/gala/potential/potential/builtin/special.html
            'nucleus': {
                'function': hernquist_acceleration,
                'params': {
                    'm': 1.8142e9, #taken from gala
                    'a': 0.0688867 
                    }
            },

            'bulge': {
                'function': hernquist_acceleration,
                'params': {
                    'm':5e9, #taken from gala
                    'a':1.0
                }
            },

            # for more accurate modeling, we use model both the MW thin and thin disks.
            # for each disk we use the MN3 method described by Gala by treating each disk as the
            # sum of three separate miyamoto-nagai disks with distinct parameters. Thus a total of 6 miyamoto-nagai disks
            # The optimal parameters shown below are from https://arxiv.org/pdf/1502.00627 section 3.2/3.3. 

            'thin_disk_1': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 9.01e10,
                    'a': 4.27,
                    'b': 0.242
                }
            },

            'thin_disk_2': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': -5.91e10,
                    'a': 9.23,
                    'b': 0.242

                }
            },

            'thin_disk_3': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 1e10,
                    'a': 1.43,
                    'b': 0.242
                }
            },

            'thick_disk_1': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 7.88e9,
                    'a': 7.30,
                    'b':1.14
                }
            },

            'thick_disk_2': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': -4.97e9,
                    'a': 15.25,
                    'b':1.14
                }
            },

            'thick_disk_3': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 0.82e9,
                    'a': 2.02,
                    'b':1.14
                }
            },

            'halo': {
                'function': nfw_acceleration,
                'params': {
                    'm':5.5427e11,
                    'r_s': 15.626
                }
            }
        }
    
    def get_acceleration(self, pos):
        '''
        Calculates the total gravitational acceleration at a given position by summing contributions
        from galactic components. Returns a 3D acceleration vector in kpc/Myr^2

        Parameters:
            - pos: a 3D np.array of the position vector [x,y,z] in kpc
        '''
        ret = np.array([0., 0., 0.])

        for component in self.components.values():
            ret += component['function'](pos, component['params'])
        
        return ret