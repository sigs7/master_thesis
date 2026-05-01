import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import src.dynamic as dps
import src.solvers as dps_sol
import importlib
importlib.reload(dps)

def _safe_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(*args, **kwargs)

if __name__ == '__main__':
    t_start_wall = time.perf_counter()
    # Model loading and initialisation
    print("Loading model data...", flush=True)
    import casestudies.ps_data.test_WT_FMU_drivetrain_ as model_data
    model = model_data.load()
    print("Building PowerSystemModel...", flush=True)
    ps = dps.PowerSystemModel(model=model)

    # UIC p_ref for power flow (FMU provides it during dynamics via connection)
    uic_model = ps.vsc['UIC_sig']
    uic_model.par['p_ref'][:] = 0.0
    uic_model.par['q_ref'][:] = 0.0

    t0 = time.perf_counter()
    print("Running power flow...", flush=True)
    ps.power_flow()
    print(f"Power flow done in {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    print("Initializing dynamic simulation (init_dyn_sim)...", flush=True)
    ps.init_dyn_sim()
    print(f"init_dyn_sim done in {time.perf_counter()-t0:.2f}s", flush=True)
    x0 = ps.x0.copy()
    v0 = ps.v0.copy()

    fmu_models = [mdl for mdl in ps.dyn_mdls if hasattr(mdl, 'step_fmu')]
    t = 0.0
    result_dict = defaultdict(list)
    # Allow quick A/B interface tests without changing the file.
    # Example: set FMU_T_END=20 to run 20 seconds.xx
    t_end = float(os.getenv('FMU_T_END', '240.0'))
    dt = 0.01
    # Use dt=0.01 to match OpenFAST FMU (canHandleVariableCommunicationStepSize=false)
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0.0, x0, t_end, max_step=dt)
    # Keep explicit current state variables (used for FMU-first co-simulation ordering).
    x = x0
    v = v0

    # Runtime storage (all FMU outputs from modelDescription.xml)
    P_e_stored = []
    P_ref_stored = []
    P_e_uic_pu_stored = []
    P_ref_uic_pu_stored = []
    v_bus = []
    fmu_outputs_stored = []
    # Also store commanded electrical torque sent to FMU (from FMUtoUICdrivetrain)
    Te_cmd_pu_stored = []
    Te_cmd_kNm_stored = []
    # Store the torque value written to FMU input GenSpdOrTrq (debugging, kN·m)
    GenSpdOrTrq_set_kNm_stored = []
    # Store the effective omega_m measurement used by the wrapper (pu)
    omega_m_pu_meas_stored = []
    # Store scaled power inputs written to the FMU (debugging)
    GenPwr_set_kW_stored = []
    ElecPwrCom_set_kW_stored = []
    GenPwr_readback_kW_stored = []
    GenSpdOrTrq_readback_kNm_stored = []
    ElecPwrCom_readback_kW_stored = []

    # Initial point
    t0 = time.perf_counter()
    print("Solving algebraic equations at t=0...", flush=True)
    v0 = ps.solve_algebraic(0.0, x0)
    print(f"solve_algebraic(t=0) done in {time.perf_counter()-t0:.2f}s", flush=True)
    result_dict['Global', 't'].append(0.0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = ps.sys_data['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    # Short circuit parameters (modify reduced Ybus diagonal at the chosen bus)
    sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]
    run_sc = False
    t_sc = 60.
    t_sc_dur = 0.05
    y_sc = 1e6

    P_e_uic = uic_model.p_e(x0, v0)[0]
    P_ref_uic = uic_model.p_ref(x0, v0)[0]
    P_e_uic_pu_stored.append(P_e_uic)
    P_ref_uic_pu_stored.append(P_ref_uic)
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

    fmu_mdl = fmu_models[0] if fmu_models else None
    if fmu_mdl and hasattr(fmu_mdl, 'get_all_fmu_outputs'):
        d0 = fmu_mdl.get_all_fmu_outputs()
        # Keep FMU-reported time, but align exported Time with TOPS time vector.
        if 'Time' in d0:
            d0 = dict(d0)
            d0['Time_fmu'] = d0.get('Time')
            d0['Time'] = float(t)
        fmu_outputs_stored.append(d0)
    if fmu_mdl is not None and hasattr(fmu_mdl, '_Te_pu_cmd'):
        te_pu = float(fmu_mdl._Te_pu_cmd) if fmu_mdl._Te_pu_cmd is not None else np.nan
        Te_cmd_pu_stored.append(te_pu)
        if hasattr(fmu_mdl, '_T_base_Nm') and np.isfinite(te_pu):
            T_base_Nm = float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0])
            Te_cmd_kNm_stored.append(te_pu * T_base_Nm / 1e3)
        else:
            Te_cmd_kNm_stored.append(np.nan)
    else:
        Te_cmd_pu_stored.append(np.nan)
        Te_cmd_kNm_stored.append(np.nan)
    if fmu_mdl is not None and hasattr(fmu_mdl, '_gen_spdortrq_kNm_set'):
        GenSpdOrTrq_set_kNm_stored.append(
            float(fmu_mdl._gen_spdortrq_kNm_set) if fmu_mdl._gen_spdortrq_kNm_set is not None else np.nan
        )
    else:
        GenSpdOrTrq_set_kNm_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_omega_m_pu_meas'):
        omega_m_pu_meas_stored.append(
            float(fmu_mdl._omega_m_pu_meas) if fmu_mdl._omega_m_pu_meas is not None else np.nan
        )
    else:
        omega_m_pu_meas_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_genpwr_kW_set'):
        GenPwr_set_kW_stored.append(
            float(fmu_mdl._genpwr_kW_set) if fmu_mdl._genpwr_kW_set is not None else np.nan
        )
    else:
        GenPwr_set_kW_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_elec_pwr_com_kW_last'):
        ElecPwrCom_set_kW_stored.append(
            float(fmu_mdl._elec_pwr_com_kW_last) if fmu_mdl._elec_pwr_com_kW_last is not None else np.nan
        )
    else:
        ElecPwrCom_set_kW_stored.append(np.nan)

    # Seed readback arrays at the initial point (t=0) to match the time vector length.
    GenPwr_readback_kW_stored.append(np.nan)
    GenSpdOrTrq_readback_kNm_stored.append(np.nan)
    ElecPwrCom_readback_kW_stored.append(np.nan)

    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))

    # Simulation loop
    while t < t_end:
        sys.stdout.write("\r%d%%" % int(t / t_end * 100))

        # Short circuit (apply at UIC terminal bus)
        if run_sc and t_sc <= t <= (t_sc + t_sc_dur):
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = y_sc
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # Step the FMU after the network/DAE step (more stable explicit coupling).
        for mdl in fmu_models:
            mdl.step_fmu(x, v, t, dt)

        result_dict['Global', 't'].append(t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        P_e_uic = uic_model.p_e(x, v)[0]
        P_ref_uic = uic_model.p_ref(x, v)[0]
        P_e_uic_pu_stored.append(P_e_uic)
        P_ref_uic_pu_stored.append(P_ref_uic)
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

        if fmu_mdl and hasattr(fmu_mdl, 'get_all_fmu_outputs'):
            d = fmu_mdl.get_all_fmu_outputs()
            # Keep FMU-reported time, but align exported Time with TOPS time vector.
            if 'Time' in d:
                d = dict(d)
                d['Time_fmu'] = d.get('Time')
                d['Time'] = float(t)
            fmu_outputs_stored.append(d)

        # Store last commanded electrical torque (as computed by the wrapper during this step)
        if fmu_mdl is not None and hasattr(fmu_mdl, '_Te_pu_cmd'):
            te_pu = float(fmu_mdl._Te_pu_cmd) if fmu_mdl._Te_pu_cmd is not None else np.nan
            Te_cmd_pu_stored.append(te_pu)
            if hasattr(fmu_mdl, '_T_base_Nm') and np.isfinite(te_pu):
                T_base_Nm = float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0])
                Te_cmd_kNm_stored.append(te_pu * T_base_Nm / 1e3)
            else:
                Te_cmd_kNm_stored.append(np.nan)
        else:
            Te_cmd_pu_stored.append(np.nan)
            Te_cmd_kNm_stored.append(np.nan)
        if fmu_mdl is not None and hasattr(fmu_mdl, '_gen_spdortrq_kNm_set'):
            GenSpdOrTrq_set_kNm_stored.append(
                float(fmu_mdl._gen_spdortrq_kNm_set) if fmu_mdl._gen_spdortrq_kNm_set is not None else np.nan
            )
        else:
            GenSpdOrTrq_set_kNm_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_omega_m_pu_meas'):
            omega_m_pu_meas_stored.append(
                float(fmu_mdl._omega_m_pu_meas) if fmu_mdl._omega_m_pu_meas is not None else np.nan
            )
        else:
            omega_m_pu_meas_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_genpwr_kW_set'):
            GenPwr_set_kW_stored.append(
                float(fmu_mdl._genpwr_kW_set) if fmu_mdl._genpwr_kW_set is not None else np.nan
            )
        else:
            GenPwr_set_kW_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_elec_pwr_com_kW_last'):
            ElecPwrCom_set_kW_stored.append(
                float(fmu_mdl._elec_pwr_com_kW_last) if fmu_mdl._elec_pwr_com_kW_last is not None else np.nan
            )
        else:
            ElecPwrCom_set_kW_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_genpwr_kW_readback'):
            GenPwr_readback_kW_stored.append(
                float(fmu_mdl._genpwr_kW_readback) if fmu_mdl._genpwr_kW_readback is not None else np.nan
            )
        else:
            GenPwr_readback_kW_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_gen_spdortrq_kNm_readback'):
            GenSpdOrTrq_readback_kNm_stored.append(
                float(fmu_mdl._gen_spdortrq_kNm_readback) if fmu_mdl._gen_spdortrq_kNm_readback is not None else np.nan
            )
        else:
            GenSpdOrTrq_readback_kNm_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_elec_pwr_com_kW_readback'):
            ElecPwrCom_readback_kW_stored.append(
                float(fmu_mdl._elec_pwr_com_kW_readback) if fmu_mdl._elec_pwr_com_kW_readback is not None else np.nan
            )
        else:
            ElecPwrCom_readback_kW_stored.append(np.nan)

        v_t_uic = uic_model.v_t(x, v)[0]
        v_bus.append(np.abs(v_t_uic))

    # Terminate FMU
    for mdl in fmu_models:
        if hasattr(mdl, 'terminate_fmu'):
            mdl.terminate_fmu()

    # Convert to DataFrame and build full export
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    t_stored = result[('Global', 't')]

    # Build export DataFrame: power system + all FMU outputs + drivetrain states (already in result)
    omega_base_rpm_export = np.nan
    if fmu_mdl is not None and hasattr(fmu_mdl, 'par') and 'omega_m_rated' in fmu_mdl.par.dtype.names:
        omega_base_rpm_export = float(np.asarray(fmu_mdl.par['omega_m_rated']).ravel()[0])
    out_df = pd.DataFrame(
        {
            't': t_stored,
            # System pu on sys base (as plotted previously)
            'P_e_sys_pu': P_e_stored,
            'P_ref_sys_pu': P_ref_stored,
            # Raw UIC pu (signed) straight from the UIC model (matches what drivetrain sees via connection)
            'P_e_uic_pu_raw': P_e_uic_pu_stored,
            'P_ref_uic_pu_raw': P_ref_uic_pu_stored,
            'v_bus_pu': v_bus,
            # Speed base for converting FMU rpm signals to pu
            'omega_base_rpm': omega_base_rpm_export,
        }
    )
    if fmu_outputs_stored:
        df_fmu = pd.DataFrame(fmu_outputs_stored)
        out_df = pd.concat([out_df, df_fmu], axis=1)

    # Add torque command traces (same length as t_stored)
    out_df['Te_cmd_pu'] = np.asarray(Te_cmd_pu_stored, dtype=float)
    out_df['Te_cmd_kNm'] = np.asarray(Te_cmd_kNm_stored, dtype=float)
    out_df['GenSpdOrTrq_set_kNm'] = np.asarray(GenSpdOrTrq_set_kNm_stored, dtype=float)
    out_df['omega_m_pu_meas'] = np.asarray(omega_m_pu_meas_stored, dtype=float)
    out_df['GenPwr_set_kW'] = np.asarray(GenPwr_set_kW_stored, dtype=float)
    out_df['ElecPwrCom_set_kW'] = np.asarray(ElecPwrCom_set_kW_stored, dtype=float)
    out_df['GenPwr_readback_kW'] = np.asarray(GenPwr_readback_kW_stored, dtype=float)
    out_df['GenSpdOrTrq_readback_kNm'] = np.asarray(GenSpdOrTrq_readback_kNm_stored, dtype=float)
    out_df['ElecPwrCom_readback_kW'] = np.asarray(ElecPwrCom_readback_kW_stored, dtype=float)

    # Init-time readbacks (constant columns; useful to verify FMU latched init params)
    if fmu_mdl is not None:
        if hasattr(fmu_mdl, '_mode_write'):
            out_df['Mode_write'] = float(fmu_mdl._mode_write)
        if hasattr(fmu_mdl, '_mode_readback'):
            out_df['Mode_readback'] = float(fmu_mdl._mode_readback)
        if hasattr(fmu_mdl, '_testNr_write'):
            out_df['testNr_write'] = float(fmu_mdl._testNr_write)
        if hasattr(fmu_mdl, '_testNr_readback'):
            out_df['testNr_readback'] = float(fmu_mdl._testNr_readback)

    # Optional: implied commanded mechanical power based on FMU-reported GenSpeed (kW) and Te_cmd_kNm
    # (kN·m * rad/s = kW)
    if 'GenSpeed' in out_df.columns:
        omega_gen = out_df['GenSpeed'].to_numpy(dtype=float) * 2.0 * np.pi / 60.0
        out_df['P_cmd_kW'] = out_df['Te_cmd_kNm'].to_numpy(dtype=float) * omega_gen
        out_df['P_cmd_pu'] = out_df['P_cmd_kW'].to_numpy(dtype=float) / (uic_s_n * 1e3)
    # Logging directory (single artifact per run; always overwrite).
    log_dir = os.path.join(project_root, 'casestudies', 'dyn_sim', 'logs', 'fmu_drivetrain')
    os.makedirs(log_dir, exist_ok=True)

    out_path = os.path.join(log_dir, 'fmu_drivetrain.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path} ({len(out_df.columns)} columns)")

    # Consistent color palette used across all figures in this script.
    # Keep ordering stable: [TOPS, command/ref, OpenFAST mech, OpenFAST/derived].
    PLOT_COLORS = ['blue', '#FF1493', 'orange', 'green']

    # Save one plot per exported signal (easy debugging)
    plots_dir = os.path.join(project_root, 'casestudies', 'dyn_sim', 'logs', 'fmu_drivetrain', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for col in out_df.columns:
        if col == 't':
            continue
        series = out_df[col]
        # Skip fully-empty / all-NaN columns
        if series.isna().all():
            continue
        fig, ax = plt.subplots(1, 1, figsize=(9, 3))
        ax.plot(out_df['t'], series, linewidth=1.0, color=PLOT_COLORS[0])
        ax.set_title(col)
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(col))
        fig.savefig(os.path.join(plots_dir, f"{safe}.png"), dpi=150)
        plt.close(fig)

    # Plot: power and drivetrain states
    df = pd.DataFrame(fmu_outputs_stored) if fmu_outputs_stored else None

    fig1, axes1 = plt.subplots(3, 1, sharex=True, figsize=(9, 11))
    fig1.suptitle('Power system & drivetrain (TOPS-side)', fontsize=12)

    axes1[0].plot(t_stored, v_bus, label='|V_t| (UIC terminal, magnitude)', color=PLOT_COLORS[0], linewidth=1.5)
    axes1[0].set_ylabel('|V_t| (p.u.)')
    _safe_legend(axes1[0], loc='best', fontsize=8)
    axes1[0].grid(True, alpha=0.3)
    # Avoid confusing "1e-5 + 1" axis offset; show actual p.u. values.
    axes1[0].ticklabel_format(axis='y', style='plain', useOffset=False)

    # Drivetrain omega subplot: TOPS omega_e + OpenFAST-reported speeds (both in pu)
    key = ('FMUtoUICdrivetrain1', 'omega_e')
    if key in result.columns:
        axes1[1].plot(t_stored, result[key], label="TOPS omega_e", color=PLOT_COLORS[0], linewidth=1.2)

    # OpenFAST outputs (FMU): RotSpeed/GenSpeed are rpm -> convert to pu using omega_m_rated base (rpm)
    omega_base_rpm = None
    if fmu_mdl is not None and hasattr(fmu_mdl, 'par') and 'omega_m_rated' in fmu_mdl.par.dtype.names:
        omega_base_rpm = float(np.asarray(fmu_mdl.par['omega_m_rated']).ravel()[0])
    if omega_base_rpm is not None and omega_base_rpm > 0 and df is not None:
        if 'RotSpeed' in df.columns:
            axes1[1].plot(
                t_stored,
                df['RotSpeed'] / omega_base_rpm,
                '--',
                label='OpenFAST omega_m (RotSpeed, p.u.)',
                color=PLOT_COLORS[2],
                linewidth=1.2,
                alpha=0.9,
            )
        if 'GenSpeed' in df.columns:
            axes1[1].plot(
                t_stored,
                df['GenSpeed'] / omega_base_rpm,
                ':',
                label='OpenFAST omega_e (GenSpeed, p.u.)',
                color=PLOT_COLORS[3],
                linewidth=1.2,
                alpha=0.9,
            )

    axes1[1].set_ylabel('Omega (p.u.)')
    _safe_legend(axes1[1], loc='best', fontsize=8)
    axes1[1].grid(True, alpha=0.3)

    # Drivetrain twist subplot
    key = ('FMUtoUICdrivetrain1', 'theta_s')
    if key in result.columns:
        axes1[2].plot(t_stored, result[key], label="TOPS theta_s", color=PLOT_COLORS[0], linewidth=1.2)
    axes1[2].set_ylabel('Shaft twist theta_s (p.u.)')
    axes1[2].set_xlabel('Time (s)')
    _safe_legend(axes1[2], loc='best', fontsize=8)
    axes1[2].grid(True, alpha=0.3)

    fig1.tight_layout()

    # Figure 1b: OpenFAST / DLL-interface signals (for debugging who regulates speed/torque)
    fig1b, axes1b = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    fig1b.suptitle('OpenFAST controller signals (FMU outputs)', fontsize=12)

    if df is not None:
        # Pitch
        if 'BldPitch1' in df.columns:
            axes1b[0].plot(t_stored, df['BldPitch1'], label='Blade pitch 1 (deg)', color=PLOT_COLORS[0], linewidth=1.2)
        axes1b[0].set_ylabel('Pitch (deg)')
        _safe_legend(axes1b[0], loc='best', fontsize=8)
        axes1b[0].grid(True, alpha=0.3)

        # Speed reference and measured speeds (all in p.u. on omega_base)
        if omega_base_rpm is not None and omega_base_rpm > 0:
            if 'RefGenSpd' in df.columns:
                axes1b[1].plot(
                    t_stored,
                    df['RefGenSpd'] / omega_base_rpm,
                    label='RefGenSpd (controller ref, p.u.)',
                    color=PLOT_COLORS[1],
                    linewidth=1.2,
                )
            if 'GenSpeed' in df.columns:
                axes1b[1].plot(
                    t_stored,
                    df['GenSpeed'] / omega_base_rpm,
                    '--',
                    label='GenSpeed (meas, p.u.)',
                    color=PLOT_COLORS[3],
                    linewidth=1.0,
                )
            if 'RotSpeed' in df.columns:
                axes1b[1].plot(
                    t_stored,
                    df['RotSpeed'] / omega_base_rpm,
                    ':',
                    label='RotSpeed (meas, p.u.)',
                    color=PLOT_COLORS[2],
                    linewidth=1.0,
                )
        axes1b[1].set_ylabel('Speed (p.u.)')
        _safe_legend(axes1b[1], loc='best', fontsize=8)
        axes1b[1].grid(True, alpha=0.3)

        # Torques (kN·m). Note: GenSpdOrTrq_set_kNm is the *exact* value sent to the FMU input
        # and may have an opposite sign convention to OpenFAST-reported torques.
        if 'GenTq' in df.columns:
            axes1b[2].plot(t_stored, df['GenTq'], label='GenTq (kN·m)', color=PLOT_COLORS[3], linewidth=1.2)
        if 'HSShftTq' in df.columns:
            axes1b[2].plot(
                t_stored, df['HSShftTq'], '--', label='HSShftTq (kN·m)', color=PLOT_COLORS[2], linewidth=1.0
            )
        if 'GenSpdOrTrq_set_kNm' in out_df.columns:
            axes1b[2].plot(
                t_stored,
                out_df['GenSpdOrTrq_set_kNm'],
                ':',
                label='GenSpdOrTrq sent (kN·m)',
                color=PLOT_COLORS[1],
                linewidth=1.0,
                alpha=0.9,
            )
        if 'Te_cmd_kNm' in out_df.columns:
            axes1b[2].plot(
                t_stored,
                out_df['Te_cmd_kNm'],
                '-.',
                label='Te_cmd (TOPS, kN·m)',
                color='black',
                linewidth=1.0,
                alpha=0.8,
            )
        axes1b[2].set_ylabel('Torque (kN·m)')
        axes1b[2].set_xlabel('Time (s)')
        _safe_legend(axes1b[2], loc='best', fontsize=8)
        axes1b[2].grid(True, alpha=0.3)

    fig1b.tight_layout()

    # Figure 2: power + torques (TOPS-side drivetrain)
    fig2, axes2 = plt.subplots(2, 1, sharex=True, figsize=(9, 8))
    fig2.suptitle('Power & torques (TOPS-side drivetrain)', fontsize=12)

    # Power (system side, already in sys pu in this script)
    axes2[0].plot(t_stored, out_df['P_e_sys_pu'], label='P_e (sys pu)', color=PLOT_COLORS[1], linewidth=1.5)
    axes2[0].plot(t_stored, out_df['P_ref_sys_pu'], '--', label='P_ref (sys pu)', color=PLOT_COLORS[0], linewidth=1.5)
    axes2[0].set_ylabel('Power (p.u.)')
    _safe_legend(axes2[0], loc='best', fontsize=8)
    axes2[0].grid(True, alpha=0.3)

    # Torques (all in local pu on drivetrain base)
    if fmu_mdl is not None and df is not None:
        eff = float(getattr(fmu_mdl, '_efficiency', 1.0))
        if not np.isfinite(eff) or eff <= 0.0:
            eff = 1.0

        # Shaft torque from TOPS states: T_shaft = K*theta_s + D*(omega_m-omega_e) (pu)
        if hasattr(fmu_mdl, 'par') and 'K' in fmu_mdl.par.dtype.names and 'D' in fmu_mdl.par.dtype.names:
            theta_key = ('FMUtoUICdrivetrain1', 'theta_s')
            omega_e_key = ('FMUtoUICdrivetrain1', 'omega_e')
            if (
                theta_key in result.columns
                and omega_e_key in result.columns
                and 'omega_m_pu_meas' in out_df.columns
            ):
                K_pu = float(np.asarray(fmu_mdl.par['K']).ravel()[0])
                D_pu = float(np.asarray(fmu_mdl.par['D']).ravel()[0])
                theta_s_arr = result[theta_key].to_numpy(dtype=float)
                omega_e_arr = result[omega_e_key].to_numpy(dtype=float)
                omega_m_arr = out_df['omega_m_pu_meas'].to_numpy(dtype=float)
                Tshaft_pu = K_pu * theta_s_arr + D_pu * (omega_m_arr - omega_e_arr)
                axes2[1].plot(
                    t_stored,
                    Tshaft_pu,
                    label='T_shaft (TOPS)',
                    color=PLOT_COLORS[0],
                    linewidth=1.4,
                )

        # Mechanical torque output from OpenFAST (HSShftTq in kN·m) -> pu on same base
        if hasattr(fmu_mdl, '_T_base_Nm') and fmu_mdl._T_base_Nm and 'HSShftTq' in df.columns:
            T_base_Nm = float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0])
            Tmech_of_pu = (df['HSShftTq'].to_numpy(dtype=float) * 1e3) / T_base_Nm
            axes2[1].plot(
                t_stored,
                Tmech_of_pu,
                label='T_mech (OpenFAST HSShftTq)',
                color=PLOT_COLORS[2],
                linewidth=1.2,
                alpha=0.9,
            )

        # Torque implied by grid electrical power:
        # - Electrical output torque:   T_e_out = P_e/omega_e
        # - Torque used by FMUtoUICdrivetrain (matches its current implementation):
        #     Te_cmd = P_e/(efficiency*omega_e)
        S_n_loc = float(np.asarray(fmu_mdl.par['S_n']).ravel()[0]) if hasattr(fmu_mdl, 'par') and 'S_n' in fmu_mdl.par.dtype.names else None
        if S_n_loc is not None and S_n_loc > 0:
            # P_e_stored is in sys pu: P_e_stored = P_e_uic_pu * (S_n_uic/sys_s_n)
            P_e_uic_pu_arr = np.asarray(P_e_stored, dtype=float) * (sys_s_n / uic_s_n)
            Pe_loc_pu_arr = P_e_uic_pu_arr * (uic_s_n / float(S_n_loc))
            omega_e_key = ('FMUtoUICdrivetrain1', 'omega_e')
            if omega_e_key in result.columns:
                omega_e_arr = result[omega_e_key].to_numpy(dtype=float)
                # Match drivetrain model logic exactly: divide if omega != 0 else 0
                Te_out_pu = np.zeros_like(omega_e_arr, dtype=float)
                np.divide(Pe_loc_pu_arr, omega_e_arr, out=Te_out_pu, where=(omega_e_arr != 0.0))
                axes2[1].plot(
                    t_stored,
                    Te_out_pu,
                    '--',
                    label='T_e_out (grid: P_e/omega_e)',
                    color=PLOT_COLORS[3],
                    linewidth=1.2,
                    alpha=0.9,
                )
                # Use the logged command from the wrapper (this is what was actually sent to the FMU)
                # to avoid any base/sign mismatch from recomputing it here.
                if 'Te_cmd_pu' in out_df.columns:
                    axes2[1].plot(
                        t_stored,
                        out_df['Te_cmd_pu'].to_numpy(dtype=float),
                        '--',
                        label='T_e_cmd (logged from wrapper)',
                        color=PLOT_COLORS[1],
                        linewidth=1.2,
                        alpha=0.9,
                    )

    axes2[1].set_ylabel('Torque (p.u. on WT base)')
    axes2[1].set_xlabel('Time (s)')
    _safe_legend(axes2[1], loc='best', fontsize=8)
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    print(f"\nSimulation took {time.perf_counter() - t_start_wall:.2f} seconds.")
    plt.show()