import pandas as pd
import numpy as np
import astropy.units as u

class TrajectoryInterpolator:
    """
    A class to load and linearly interpolate trajectory data for the MW and LMC
    
    The input data file is expected to be a whitespace-delimited text file with four
    columns: time, x, y, and z, and no header.
    """
    def __init__(self, file_path: str):
        self.GYR_TO_MYR = 1000.0

        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        col_names = ['time', 'x', 'y', 'z']
        
        df = pd.read_csv(
            self.file_path,
            delim_whitespace=True,
            header=None,
            names=col_names
        )
        #convert gyr to myr
        df['time'] *= self.GYR_TO_MYR
        #ensure sorted
        df = df.sort_values(by='time').reset_index(drop=True)
        
        return df

    def get_position(self, t: float) -> np.ndarray:
        x = np.interp(t, self.data['time'], self.data['x'])
        y = np.interp(t, self.data['time'], self.data['y'])
        z = np.interp(t, self.data['time'], self.data['z'])
        
        return np.array([x, y, z])