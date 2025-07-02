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
        - params: A dictionary of the parameters. 'm' for mass in Suns, 'l' for scale length (kpc), 'h' for scale height
    """
    pass

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
    
    rho = params['rho0']
    a = params['a']
    factored_term = -4 * PI * G_KPC_MYR * rho * (a**3) / (r**3)
    bracket_term = r / (r + a) - np.log(1 + r / a)

    return factored_term * bracket_term * pos



class MWPotential:
    def __init__(self):

        # define component values
        rho0_halo = None #FIND THESE VALUES LATER
        scale_radius_halo = None
    
        self.components = {
            # initialize parameters for the components
            'nucleus': {
                'function': hernquist_acceleration,
                'params': {
                    'm': 4.297e6, # mass of SGR A* in sun mass
                    'a': 0.00001 # is this right?? -- trying to treat as point mass
                    }
            },

            'bulge': {
                'function': hernquist_acceleration,
                'params': {
                    # NEED MASS AND SCALE RADIUS OF BULGE
                }
            },

            # for more accurate modeling, we use model both the MW thin and thin disks.
            # for each disk we use the MN3 method described by Gala by treating each disk as the
            # sum of three separate miyamoto-nagai disks with distinct parameters. Thus a total of 6 miyamoto-nagai disks
            # The optimal parameters shown below are from https://arxiv.org/pdf/1502.00627 section 3. 

            'thin_disk_1': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 9.01e10,
                    'a': 4.27
                }
            },

            'thin_disk_2': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': -5.91e10,
                    'a': 9.23
                }
            },

            'thin_disk_3': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 1e10,
                    'a': 1.43
                }
            },

            'thick_disk_1': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 7.88e9,
                    'a': 7.30
                }
            },

            'thick_disk_2': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': -4.97e9,
                    'a': 15.25
                }
            },

            'thick_disk_3': {
                'function': miyamoto_nagai_acceleration,
                'params': {
                    'm': 0.82e9,
                    'a': 2.02
                }
            },

            'halo': {
                'function': nfw_acceleration,
                'params': {
                    'rho0': rho0_halo,
                    'a': scale_radius_halo
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