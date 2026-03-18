# Setup (Wind turbine FMU simulation)

1. **Clone the repository** and `cd` into the project root (`prosjektoppgave-tops-copy-ext`).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place `fast.fmu`** in the project root. The OpenFAST FMU must be built separately and placed here.

4. **Run the simulation** from the project root:
   ```bash
   python casestudies/dyn_sim/test_WT_FMU_sim.py
   ```

   The script sets the working directory to the project root automatically. The FMU reads OpenFAST inputs from `OpenFAST/test1002` (IEA 15 MW turbine). Results are written to `test_WT_FMU_sim_results.csv` in the project root.

# TOPS (**T**iny **O**pen **P**ower System **S**imulator)

For use in the course TET4180 at NTNU. Huge thanks to hallvard-h for providing us with this lightweight simluation tool.

From hallvard-h's readme:

This is a package for performing dynamic power system simulations in Python. The aim is to provide a simple and lightweight tool which is easy to install, run and modify, to be used by researchers and in education. Performance is not the main priority. The only dependencies are numpy, scipy, pandas and matplotlib (the core functionality only uses numpy and scipy).

The package is being developed as part of ongoing research, and thus contains experimental features. Use at your own risk!

Some features:
- Newton-Rhapson power flow
- Dynamic time domain simulation (RMS/phasor approximation)
- Linearization, eigenvalue analysis/modal analysis


# Citing (TOPS)
If you use this code for your research, please cite [this paper](https://arxiv.org/abs/2101.02937).

# Example notebooks (TOPS)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

# Contact (TOPS)
[Hallvar Haugdal](mailto:hallvhau@gmail.com)
