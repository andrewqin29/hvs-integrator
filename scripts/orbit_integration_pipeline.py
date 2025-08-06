import sys
import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, chi2
from tqdm import tqdm
import seaborn as sns

# disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Setup Project Paths ---
script_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from hvs_orbital_kinematics.potentials import MWPotential
from hvs_orbital_kinematics.integrators import leapfrog_step_time_varying
from hvs_orbital_kinematics.interpolators import TrajectoryInterpolator
from astropy import units as u

# --- Constants ---
KM_S_TO_KPC_MYR = (u.km / u.s).to(u.kpc / u.Myr)

# --- Helper Functions ---
def find_closest_passages(initial_pos, initial_vel, potential, times, dt):
    """Integrates a single orbit backward and finds the closest passage to the MW and LMC."""
    pos, vel = initial_pos, initial_vel
    min_dist_sq_mw, min_dist_sq_lmc = np.inf, np.inf
    closest_pos_mw, closest_pos_lmc = np.full(3, np.nan), np.full(3, np.nan)

    for t in times:
        pos, vel = leapfrog_step_time_varying(pos, vel, t, dt, potential)
        mw_center = potential.mw_interpolator.get_position(t)
        lmc_center = potential.lmc_interpolator.get_position(t)
        dist_sq_mw = np.sum((pos - mw_center)**2)
        dist_sq_lmc = np.sum((pos - lmc_center)**2)
        
        if dist_sq_mw < min_dist_sq_mw:
            min_dist_sq_mw = dist_sq_mw
            closest_pos_mw = pos - mw_center
        if dist_sq_lmc < min_dist_sq_lmc:
            min_dist_sq_lmc = dist_sq_lmc
            closest_pos_lmc = pos - lmc_center
            
    return closest_pos_mw, closest_pos_lmc

def calculate_p_value(passage_points):
    """Calculates the p-value for an origin hypothesis using the Mahalanobis distance."""
    valid_points = passage_points[~np.isnan(passage_points).any(axis=1)]
    if len(valid_points) < 10: return np.nan
    mean_vec = np.mean(valid_points, axis=0)
    cov_matrix = np.cov(valid_points, rowvar=False)
    point_to_test = np.array([0., 0., 0.])
    delta = point_to_test - mean_vec
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahalanobis_sq = delta.T @ inv_cov_matrix @ delta
    except np.linalg.LinAlgError:
        return np.nan
    return chi2.sf(mahalanobis_sq, df=3)

def main():
    print("--- HVS ORBIT INTEGRATION PIPELINE ---")
    
    # --- Load Base Data ---
    data_dir = os.path.join(project_root, 'data')
    hvs_data_path = os.path.join(data_dir, 'processed', '6d_cartesian_data.csv')
    covariance_data_path = os.path.join(data_dir, 'processed', '6d_cartesian_covariance.csv')
    cartesian_df = pd.read_csv(hvs_data_path)
    covariance_df = pd.read_csv(covariance_data_path)
    print("Successfully loaded HVS and covariance data.\n")

    # Define the 8 trajectory models to simulate
    trajectories = [
        {'is_radial': True, 'mass_model': 1}, {'is_radial': True, 'mass_model': 2},
        {'is_radial': True, 'mass_model': 3}, {'is_radial': True, 'mass_model': 4},
        {'is_radial': False, 'mass_model': 1}, {'is_radial': False, 'mass_model': 2},
        {'is_radial': False, 'mass_model': 3}, {'is_radial': False, 'mass_model': 4}
    ]

    for traj in trajectories:
        # --- PARAMETER SETUP FOR CURRENT TRAJECTORY MODEL ---
        is_radial = traj['is_radial']
        mass_model = traj['mass_model']
        
        LMC_PARAMS = {
            1: {'m_vir': 8e10,  'r_a': 10.4}, 
            2: {'m_vir': 10e10, 'r_a': 12.7}, 
            3: {'m_vir': 18e10, 'r_a': 20.0}, 
            4: {'m_vir': 25e10, 'r_a': 25.2}  
        }
        lmc_params = LMC_PARAMS.get(mass_model, LMC_PARAMS[3])

        model_type_str = 'radial' if is_radial else 'isotropic'
        folder_name = f"{model_type_str}_lmc_{mass_model}"
        output_dir = os.path.join(project_root, 'data', 'processed', 'mass_corrected_origin_analysis', folder_name)
        
        # --- Clean up old data in the directory before running ---
        if os.path.exists(output_dir):
            print(f"Cleaning previous run data in: {folder_name}")
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(output_dir)
        
        print("="*60)
        print(f"STARTING SIMULATION FOR TRAJECTORY MODEL: {folder_name}")
        print(f"Output will be saved to: {output_dir}")
        print("="*60)

        try:
            b_val = 1 if is_radial else 0
            mw_orbit_path = os.path.join(data_dir, 'raw', 'trajectories', model_type_str, f'GC21M{mass_model}b{b_val}_orbit_mw.txt')
            lmc_orbit_path = os.path.join(data_dir, 'raw', 'trajectories', model_type_str, f'GC21M{mass_model}b{b_val}_orbit_lmc.txt')
        except FileNotFoundError:
            print(f"ERROR: Trajectory files for {folder_name} not found. Skipping.")
            continue

        potential = MWPotential(mw_orbit_path=mw_orbit_path, lmc_orbit_path=lmc_orbit_path)
        potential.set_lmc_hernquist(m=lmc_params['m_vir'], c=lmc_params['r_a'])
        print(f"LMC parameters set to M={potential.lmc_component['params']['m']:.2e} M_sun, c={potential.lmc_component['params']['c']:.2f} kpc.")

        # --- ORBIT INTEGRATION ---
        N_SAMPLES = 100
        N_STEPS = 800    
        DT = -0.5        
        times = np.arange(0, N_STEPS * DT, DT)
        all_hvs_ids = sorted(cartesian_df['HVS'].unique())
        all_results = []

        mw_pos_t0 = potential.mw_interpolator.get_position(0)
        mw_vel_t0 = potential.mw_interpolator.get_velocity(0)
        mw_state_vector_t0 = np.concatenate([mw_pos_t0, mw_vel_t0])

        for hvs_id in all_hvs_ids:
            print(f"\nProcessing HVS {hvs_id} for model {folder_name}...")
            # (Integration logic remains the same)
            star_means_df = cartesian_df[cartesian_df['HVS'] == hvs_id]
            star_cov_df = covariance_df[covariance_df['HVS'] == hvs_id]
            mean_vec = star_means_df[['x', 'y', 'z', 'u', 'v', 'w']].values.flatten()
            mean_vec[3:] *= KM_S_TO_KPC_MYR
            cov_matrix = np.zeros((6, 6))
            var_names = ['x', 'y', 'z', 'u', 'v', 'w']
            for i in range(6):
                for j in range(i, 6):
                    cov_val = star_cov_df[f'cov_{var_names[i]}{var_names[j]}'].iloc[0]
                    scale_i = KM_S_TO_KPC_MYR if i >= 3 else 1
                    scale_j = KM_S_TO_KPC_MYR if j >= 3 else 1
                    cov_matrix[i, j] = cov_matrix[j, i] = cov_val * scale_i * scale_j
            relative_conditions = np.random.multivariate_normal(mean_vec, cov_matrix, N_SAMPLES)
            initial_conditions = relative_conditions + mw_state_vector_t0
            closest_passages_mw = []
            closest_passages_lmc = []
            for i in tqdm(range(N_SAMPLES), desc=f"Integrating for HVS {hvs_id}"):
                initial_pos, initial_vel = initial_conditions[i, :3], initial_conditions[i, 3:]
                cp_mw, cp_lmc = find_closest_passages(initial_pos, initial_vel, potential, times, DT)
                closest_passages_mw.append(cp_mw)
                closest_passages_lmc.append(cp_lmc)
            p_value_mw = calculate_p_value(np.array(closest_passages_mw))
            p_value_lmc = calculate_p_value(np.array(closest_passages_lmc))
            all_results.append({
                'id': hvs_id,
                'mw_passages': np.array(closest_passages_mw),
                'lmc_passages': np.array(closest_passages_lmc),
                'p_mw': p_value_mw,
                'p_lmc': p_value_lmc
            })
        
        print(f"\n--- SIMULATION COMPLETE for {folder_name} ---")
        
        # --- SAVING DATA FOR CURRENT MODEL ---
        summary_data = [{'hvs_id': res['id'],
                         'p-value of MW origin': f"{res['p_mw']:.4f}" if not np.isnan(res['p_mw']) else 'NaN',
                         'p-value of LMC origin': f"{res['p_lmc']:.4f}" if not np.isnan(res['p_lmc']) else 'NaN'}
                        for res in all_results]
        summary_df = pd.DataFrame(summary_data)
        summary_filename = "p_value_summary.txt"
        summary_filepath = os.path.join(output_dir, summary_filename)
        summary_df.to_csv(summary_filepath, sep='\t', index=False)
        print(f"P-value summary saved to: {summary_filepath}")

    # --- FINAL CROSS-MODEL ANALYSIS (after all trajectories are run) ---
    print("\n\n" + "="*60)
    print("ALL TRAJECTORY SIMULATIONS COMPLETE. STARTING CROSS-MODEL ANALYSIS...")
    print("="*60)
    
    analysis_base_dir = os.path.join(project_root, 'data', 'processed', 'mass_corrected_origin_analysis')
    
    # --- Clean up old analysis plots ---
    for plot_file in ['pvalue_heatmap_lmc.png', 'pvalue_dotplot_lmc.png', 'pvalue_heatmap_mw.png', 'pvalue_dotplot_mw.png']:
        old_plot_path = os.path.join(analysis_base_dir, plot_file)
        if os.path.exists(old_plot_path):
            os.remove(old_plot_path)
            print(f"Removed old analysis plot: {plot_file}")

    all_pvalue_dfs = []
    for folder in os.listdir(analysis_base_dir):
        summary_file = os.path.join(analysis_base_dir, folder, 'p_value_summary.txt')
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file, sep='\t')
            df = df.rename(columns={
                'p-value of MW origin': f'p_mw_{folder}',
                'p-value of LMC origin': f'p_lmc_{folder}'
            })
            all_pvalue_dfs.append(df)

    if not all_pvalue_dfs:
        print("No summary files found for cross-model analysis. Exiting.")
        return

    master_df = all_pvalue_dfs[0]
    for df in all_pvalue_dfs[1:]:
        master_df = pd.merge(master_df, df, on='hvs_id', how='outer')

    # --- Process LMC Data ---
    lmc_cols = [col for col in master_df.columns if 'p_lmc' in col]
    lmc_analysis_results = [{'HVS ID': row['hvs_id'],
                             'Mean LMC p-value': row[lmc_cols].astype(float).mean(),
                             'Std Dev LMC p-value': row[lmc_cols].astype(float).std()}
                            for _, row in master_df.iterrows()]
    lmc_analysis_df = pd.DataFrame(lmc_analysis_results)

    # --- Process MW Data ---
    mw_cols = [col for col in master_df.columns if 'p_mw' in col]
    mw_analysis_results = [{'HVS ID': row['hvs_id'],
                            'Mean MW p-value': row[mw_cols].astype(float).mean(),
                            'Std Dev MW p-value': row[mw_cols].astype(float).std()}
                           for _, row in master_df.iterrows()]
    mw_analysis_df = pd.DataFrame(mw_analysis_results)
    
    print("\n--- Cross-Model Statistical Analysis (LMC) ---")
    print(lmc_analysis_df.to_string(index=False))
    print("\n--- Cross-Model Statistical Analysis (MW) ---")
    print(mw_analysis_df.to_string(index=False))

    # --- Visualization 1: LMC P-Value Heatmap ---
    fig_heat_lmc, ax_heat_lmc = plt.subplots(figsize=(12, 9))
    heatmap_data_lmc = master_df.set_index('hvs_id')[lmc_cols]
    heatmap_data_lmc.columns = [col.replace('p_lmc_', '') for col in lmc_cols]
    sns.heatmap(heatmap_data_lmc, annot=True, cmap="viridis", fmt=".2f", linewidths=.5, ax=ax_heat_lmc)
    ax_heat_lmc.set_title('LMC Origin P-Value Across Trajectory Models', fontsize=16)
    ax_heat_lmc.set_xlabel('Trajectory Model', fontsize=12)
    ax_heat_lmc.set_ylabel('HVS ID', fontsize=12)
    plt.tight_layout()
    heatmap_path_lmc = os.path.join(analysis_base_dir, 'pvalue_heatmap_lmc.png')
    fig_heat_lmc.savefig(heatmap_path_lmc, dpi=200)
    print(f"\nLMC Heatmap saved to: {heatmap_path_lmc}")
    plt.show()

    # --- Visualization 2: LMC Dot Plot with Error Bars ---
    fig_dot_lmc, ax_dot_lmc = plt.subplots(figsize=(12, 8))
    ax_dot_lmc.errorbar(lmc_analysis_df['HVS ID'].astype(str), lmc_analysis_df['Mean LMC p-value'], yerr=lmc_analysis_df['Std Dev LMC p-value'], 
                        fmt='o', capsize=5, color='dodgerblue', ecolor='gray', markersize=8, linestyle='None', label='Mean LMC p-value')
    ax_dot_lmc.set_xlabel('HVS ID', fontsize=12)
    ax_dot_lmc.set_ylabel('Mean LMC Origin p-value', fontsize=12)
    ax_dot_lmc.set_title('Mean LMC p-value Across 8 Trajectory Models', fontsize=16)
    ax_dot_lmc.tick_params(axis='x', rotation=45)
    ax_dot_lmc.set_ylim(0, 1)
    ax_dot_lmc.legend()
    ax_dot_lmc.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    dotplot_path_lmc = os.path.join(analysis_base_dir, 'pvalue_dotplot_lmc.png')
    fig_dot_lmc.savefig(dotplot_path_lmc, dpi=200)
    print(f"LMC Dot plot saved to: {dotplot_path_lmc}")
    plt.show()
    
    # --- Visualization 3: MW P-Value Heatmap ---
    fig_heat_mw, ax_heat_mw = plt.subplots(figsize=(12, 9))
    heatmap_data_mw = master_df.set_index('hvs_id')[mw_cols]
    heatmap_data_mw.columns = [col.replace('p_mw_', '') for col in mw_cols]
    sns.heatmap(heatmap_data_mw, annot=True, cmap="magma", fmt=".2f", linewidths=.5, ax=ax_heat_mw)
    ax_heat_mw.set_title('MW Origin P-Value Across Trajectory Models', fontsize=16)
    ax_heat_mw.set_xlabel('Trajectory Model', fontsize=12)
    ax_heat_mw.set_ylabel('HVS ID', fontsize=12)
    plt.tight_layout()
    heatmap_path_mw = os.path.join(analysis_base_dir, 'pvalue_heatmap_mw.png')
    fig_heat_mw.savefig(heatmap_path_mw, dpi=200)
    print(f"\nMW Heatmap saved to: {heatmap_path_mw}")
    plt.show()

    # --- Visualization 4: MW Dot Plot with Error Bars ---
    fig_dot_mw, ax_dot_mw = plt.subplots(figsize=(12, 8))
    ax_dot_mw.errorbar(mw_analysis_df['HVS ID'].astype(str), mw_analysis_df['Mean MW p-value'], yerr=mw_analysis_df['Std Dev MW p-value'], 
                       fmt='o', capsize=5, color='purple', ecolor='gray', markersize=8, linestyle='None', label='Mean MW p-value')
    ax_dot_mw.set_xlabel('HVS ID', fontsize=12)
    ax_dot_mw.set_ylabel('Mean MW Origin p-value', fontsize=12)
    ax_dot_mw.set_title('Mean MW p-value Across 8 Trajectory Models', fontsize=16)
    ax_dot_mw.tick_params(axis='x', rotation=45)
    ax_dot_mw.set_ylim(0, 1)
    ax_dot_mw.legend()
    ax_dot_mw.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    dotplot_path_mw = os.path.join(analysis_base_dir, 'pvalue_dotplot_mw.png')
    fig_dot_mw.savefig(dotplot_path_mw, dpi=200)
    print(f"MW Dot plot saved to: {dotplot_path_mw}")
    plt.show()

if __name__ == '__main__':
    main()
