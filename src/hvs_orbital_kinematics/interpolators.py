import pandas as pd
import numpy as np
import astropy.units as u

class TrajectoryInterpolator:
    """
    A class to load and linearly interpolate trajectory data for the MW and LMC.
    """
    def __init__(self, file_path: str):
        
        self.GYR_TO_MYR = 1000.0
        self.KM_S_TO_KPC_MYR = (u.km / u.s).to(u.kpc / u.Myr)

        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:    
        col_names = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        
        df = pd.read_csv(
            self.file_path,
            delim_whitespace=True,
            header=None,
            names=col_names
        )

        df['time'] *= self.GYR_TO_MYR
        df['vx'] *= self.KM_S_TO_KPC_MYR
        df['vy'] *= self.KM_S_TO_KPC_MYR
        df['vz'] *= self.KM_S_TO_KPC_MYR
        
        df = df.sort_values(by='time').reset_index(drop=True)
        
        return df

    def get_position(self, t: float) -> np.ndarray:
        x = np.interp(t, self.data['time'], self.data['x'])
        y = np.interp(t, self.data['time'], self.data['y'])
        z = np.interp(t, self.data['time'], self.data['z'])
        
        return np.array([x, y, z])

    def get_velocity(self, t: float) -> np.ndarray:
        vx = np.interp(t, self.data['time'], self.data['vx'])
        vy = np.interp(t, self.data['time'], self.data['vy'])
        vz = np.interp(t, self.data['time'], self.data['vz'])
        
        return np.array([vx, vy, vz])
