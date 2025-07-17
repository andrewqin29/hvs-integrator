# interpolation on LMC and MW trajectory data
import os

file_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(file_dir, os.pardir))
mw_orbit_path = os.path.join(project_root, 'data', 'raw', 'trajectories', 'GC21M2b1_orbit_mw.txt')
lmc_orbit_path = os.path.join(project_root, 'data', 'raw', 'trajectories', 'GC21M3b1_orbit_lmc.txt')


