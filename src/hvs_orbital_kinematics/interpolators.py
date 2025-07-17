import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

class TrajectoryInterpolator:
    def __init__(self, file_path: str, poly_degree: int):
        self.file_path = file_path
        self.poly_degree = poly_degree
        self.data = self.load_data()
        # final coefficients after model fit
        self.params = {}
        self.fit_models()

    def load_data(self) -> pd.DataFrame:
        col_names = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        df = pd.read_csv(self.file_path, delim_whitespace=True, header=None, names=col_names)
        return df
    
    def poly_model(self, t, *coeffs) -> np.ndarray:
        return np.polyval(t, coeffs)

    def fit_model(self):
        coords_to_fit = ['x', 'y', 'z']
        initial_guess = [0.0] * (self.poly_degree + 1)

        for coord in coords_to_fit:
            popt, _ = curve_fit(
                self._poly_model,
                self.data['time'],
                self.data[coord],
                p0=initial_guess
            )
            self.params[coord] = popt
    
    def get_position(self, t: float) -> np.ndarray:
        x = self._poly_model(t, *self.params['x'])
        y = self._poly_model(t, *self.params['y'])
        z = self._poly_model(t, *self.params['z'])
        return np.array([x, y, z])
        