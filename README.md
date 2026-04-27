# Master thesis – Wind turbine power system simulation

This repository contains a TOPS-based power system simulator with two ways to model the wind turbine:

| Type | Description | Use case |
|------|-------------|----------|
| **Internal model** | Built-in `WindTurbine` (`src/dyn_models/windturbine.py`) – simplified drivetrain, MPT, Cp tables | Fast runs, no FMU, no OpenFAST |
| **FMU co-simulation** | OpenFAST FMU – full aeroelastic turbine model | High-fidelity turbine dynamics |

## Setup

1. **Clone the repository** and `cd` into the project root.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running simulations

### Internal wind turbine model

Uses the simplified `WindTurbine` model. No FMU required. Uses `wind_data/` (MPT, Cp tables).

```bash
python casestudies/dyn_sim/test_WT_sim.py
```

### FMU co-simulation (OpenFAST)

Uses the OpenFAST FMU for full turbine dynamics. Requires `OpenFAST/fast.fmu` and `OpenFAST/test1002` (IEA 15 MW).

1. Place `fast.fmu` at `OpenFAST/fast.fmu`.
2. Run:
   ```bash
   python casestudies/dyn_sim/test_WT_FMU_sim.py
   ```

   Results are written to `test_WT_FMU_sim_results.csv` in the project root.

## TOPS

This project builds on **TOPS** (Tiny Open Power System Simulator). For use in the course TET4180 at NTNU. Huge thanks to hallvard-h for this lightweight simulation tool.

Features: Newton-Raphson power flow, dynamic time domain simulation (RMS/phasor), linearization, eigenvalue analysis/modal analysis.

**Citing:** If you use this code for your research, please cite [this paper](https://arxiv.org/abs/2101.02937).

**Contact:** [Hallvar Haugdal](mailto:hallvhau@gmail.com)
