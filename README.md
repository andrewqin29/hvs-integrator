### Symplectic Orbit Integration for Tracing Hypervelocity Stars
#### Data
A collection of the 6D phase space coordinates and stellar trajectory reconstructions for select hypervelocity stars presented in the paper *Hypervelocity Stars Trace a Supermassive Black Hole in the Large Magellanic Cloud* (Han et al. 2025). Data obtained from Gaia DR3 dataset and processed via Astropy and Gala. 

#### Code
- datagen.ipynb: Pipeline that generates the 6D phase space coordinates of the 24 HVS in the Han paper. Records source_id, RA, Dec, proper motions, radial distance and associated errors. Includes basic data visualization of RA/Dec positions and proper motions in the form of a vector field plot.
- datavis.ipynb: Data visualization of collected data targeting spatial and quantitative representations.
- trajectories.ipynb: Reconstruction of stellar trajectories via various techniques: AstroPy integrated orbit calculator, leapfrop integration and general experimentation.

Paper: https://arxiv.org/abs/2502.00102
