import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

class TrajectoryInterpolator:
    def __init__(self, file_path: str, poly_degree: int = 4):
        # handle unit conversions between Garavito-Camargo data and cartesian_df data
        self.KM_S_TO_KPC_MYR = (u.km / u.s).to(u.kpc / u.Myr).value
        self.GYR_TO_MYR = 1000.0

        self.file_path = file_path
        self.poly_degree = poly_degree
        self.data = self._load_data()

        self.coeffs = {}
        self._fit_models()

    def _load_data(self) -> pd.DataFrame:
        col_names = ['time', 'x', 'y', 'z']
        df = pd.read_csv(
            self.file_path,
            delim_whitespace=True,
            header=None,
            names=col_names,
            usecols=[0, 1, 2, 3]
        )

        df['time'] *= self.GYR_TO_MYR
        return df
    
    def _fit_models(self):
        coords_to_fit = ['x', 'y', 'z']
        for coord in coords_to_fit:
            self.coeffs[coord] = np.polyfit(
                self.data['time'],
                self.data[coord],
                self.poly_degree
            )

    def get_position(self, t: float) -> np.ndarray:
        x = np.polyval(self.coeffs['x'], t)
        y = np.polyval(self.coeffs['y'], t)
        z = np.polyval(self.coeffs['z'], t)
        return np.array([x, y, z])
        