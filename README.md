# Master thesis – Wind turbine power system simulation

This repository contains a TOPS-based power system simulator with two ways to model the wind turbine:

| Type | Description | Use case |
|------|-------------|----------|
| **Internal model** | Built-in `WindTurbine` (`src/dyn_models/windturbine.py`) – simplified drivetrain, MPT, Cp tables, wind file | Fast runs, no FMU, no OpenFAST |
| **FMU co-simulation** | OpenFAST FMU – full aeroelastic turbine model | High-fidelity turbine dynamics |

## Setup

1. **Clone the repository** and `cd` into the project root (`prosjektoppgave-tops-copy-ext`).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running simulations

### Internal wind turbine model

Uses the simplified `WindTurbine` model. No FMU required. Uses `wind_data/` (MPT, Cp tables, wind `.hh` file).

```bash
cd prosjektoppgave-tops-copy-ext
python casestudies/dyn_sim/uic_sim.py
```

### FMU co-simulation (OpenFAST)

Uses the OpenFAST FMU for full turbine dynamics. Requires `fast.fmu` in the project root and `OpenFAST/test1002` (IEA 15 MW).

1. Place `fast.fmu` in `prosjektoppgave-tops-copy-ext/`.
2. Run:
   ```bash
   cd prosjektoppgave-tops-copy-ext
   python casestudies/dyn_sim/test_WT_FMU_sim.py
   ```

   Results are written to `test_WT_FMU_sim_results.csv` in the project root.

## Project structure

- `prosjektoppgave-tops-copy-ext/` – TOPS power system simulator
- `prosjektoppgave-tops-copy-ext/src/dyn_models/windturbine.py` – internal wind turbine model
- `prosjektoppgave-tops-copy-ext/wind_data/` – MPT, Cp tables, wind files (for internal model)
- `prosjektoppgave-tops-copy-ext/README.md` – TOPS documentation
