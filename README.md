# Kinematic Analysis and Orbit Integration of Hypervelocity Stars

This repository contains the data processing pipelines, analysis scripts, and visualization notebooks for a research project investigating the origins of hypervelocity stars (HVS). The primary goal is to analyze the kinematics of the 20 HVS from the HVS Survey to test the hypothesis that a significant fraction originate from the Large Magellanic Cloud (LMC) rather than the Galactic Center (Sgr A*).

The project uses observational data from the Gaia DR3 dataset, processes it into a 6D Galactocentric Cartesian frame with propagated errors via Monte Carlo methods, and provides tools for kinematic analysis and past orbit integration.

### Key Scientific References
- **Primary Reference:** Han et al. (2025), *Hypervelocity Stars Trace a Supermassive Black Hole in the Large Magellanic Cloud* ([arXiv:2502.00102](https://arxiv.org/abs/2502.00102))
- **LMC Proper Motion Data:** Kallivayalil et al. (2013), *Third-epoch Magellanic Cloud Proper Motions...* ([arXiv:1301.0832](https://arxiv.org/abs/1301.0832))

---

## Setup and Installation

To set up the local environment and run the pipelines, follow these steps. This project uses a Python virtual environment to manage dependencies.

**1. Clone the Repository:**
```bash
git clone git@github.com:andrewqin29/hvs-integrator.git
cd hvs-integrator
```

**2. Create and Activate a Virtual Environment:**
```bash
# Create the virtual environment
python3 -m venv hvs_env

# Activate the environment
# On macOS/Linux:
source hvs_env/bin/activate
```

**3. Install Dependencies:**
All required Python packages are listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```

---

## Project Workflow and Usage

The project is designed as a sequential pipeline. The scripts in the `scripts/` directory must be run first to generate the necessary data, which can then be explored in the `notebooks/`.

**Step 1: Generate Processed Data**

Run the following scripts from the command line in order. They will read the raw data, query Gaia, and perform the Monte Carlo transformation.

```bash
# Script 1: Cleans raw data and queries Gaia for full observational data
python scripts/get_gaia_data.py

# Script 2: Converts observational data to Cartesian coordinates with errors
python scripts/make_cartesian_data.py
```
Upon completion, the processed data files will be saved in the `data/processed/` directory.

**Step 2: Interactive Analysis and Visualization**

Once the data is generated, you can explore it using the Jupyter Notebooks in the `notebooks/` directory.

* **`notebooks/data_generation.ipynb`**: Contains the original exploratory code and development scratchpad for the data processing pipelines.
* **`notebooks/data_visualization.ipynb`**: Loads the final Cartesian data to produce all diagnostic plots (Aitoff projections, 3D quiver plots, histograms, etc.).
* **`notebooks/orbit_integration.ipynb`**: The development notebook for building and validating the custom Leapfrog orbit integrator.

---

## Repository Structure

The repository is organized to separate data, source code, scripts, and notebooks.

```
hvs-kinematics/
|
├── .gitignore         
├── README.md          
├── requirements.txt    # Dependencies
|
├── data/
|   ├── raw/            # Contains the original, immutable source data
|   └── processed/      # Contains all generated data files from the scripts
|
├── notebooks/          # Jupyter notebooks for exploration, visualization, and reporting
|
├── scripts/            # Finalized, runnable .py scripts for the data processing pipeline
|
└── src/
    └── hvs_orbits/     # A custom Python package for this project
        ├── __init__.py   # Makes hvs_orbits an importable package
        ├── potentials.py # Functions for defining gravitational potentials
        └── integrators.py# Contains the custom Leapfrog orbit integrator
