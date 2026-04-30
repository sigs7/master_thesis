import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


PLOT_COLORS = ['blue', '#FF1493', 'orange', 'green']

def _pick_latest_csv(script_dir: str, filename_glob: str) -> str:
    """
    Pick newest CSV (mtime) matching glob in this folder.
    """
    import glob

    pattern = os.path.join(script_dir, filename_glob)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return max(matches, key=lambda p: os.path.getmtime(p))


def _ensure_increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0:
        return t
    # If there are duplicates, keep first occurrence (np.interp requires increasing x)
    _, idx = np.unique(t, return_index=True)
    idx = np.sort(idx)
    return t[idx]


def _align_df_on_time(df: pd.DataFrame, t_target: np.ndarray, t_col: str = 't') -> pd.DataFrame:
    """
    Interpolate all numeric columns of df onto t_target.
    Non-numeric columns are dropped.
    """
    if t_col not in df.columns:
        raise KeyError(f"Missing time column '{t_col}'")
    t_src = np.asarray(df[t_col], dtype=float).ravel()
    if t_src.size == 0:
        raise ValueError("Empty time vector")

    # Keep only strictly increasing (drop duplicates)
    uniq, idx = np.unique(t_src, return_index=True)
    order = np.argsort(uniq)
    t_src = uniq[order]
    keep_rows = idx[order]
    d = df.iloc[keep_rows].copy()

    out = pd.DataFrame({t_col: np.asarray(t_target, dtype=float)})
    for col in d.columns:
        if col == t_col:
            continue
        s = d[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        y = np.asarray(s, dtype=float).ravel()
        # Guard against all-NaN
        if np.all(~np.isfinite(y)):
            continue
        # Fill NaNs by linear interpolation on source grid (simple, avoids gaps)
        good = np.isfinite(y)
        if np.any(good) and not np.all(good):
            y = np.interp(t_src, t_src[good], y[good])
        out[col] = np.interp(t_target, t_src, y)
    return out


def _pu_from_rpm(rpm: np.ndarray, omega_base_rpm: float) -> np.ndarray:
    rpm = np.asarray(rpm, dtype=float)
    if not np.isfinite(omega_base_rpm) or omega_base_rpm <= 0.0:
        return np.full_like(rpm, np.nan, dtype=float)
    return rpm / omega_base_rpm


def _plain_y(ax):
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Compare WT vs FMU result CSVs.")
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show interactive window (save PNG only).",
    )
    args = ap.parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Keep this dead simple: plot the stable "LATEST" CSV if present.
    wt_csv = os.path.join(project_root, 'casestudies', 'dyn_sim', 'logs', 'wt', 'wt_model.csv')
    fmu_csv = os.path.join(project_root, 'casestudies', 'dyn_sim', 'logs', 'fmu_drivetrain', 'fmu_drivetrain.csv')
    print(f"Using WT CSV:  {wt_csv}")
    print(f"Using FMU CSV: {fmu_csv}")

    if not os.path.exists(wt_csv):
        raise FileNotFoundError(f"Missing WT results CSV: {wt_csv}")
    if not os.path.exists(fmu_csv):
        raise FileNotFoundError(
            f"Missing FMU latest results CSV: {fmu_csv}\n"
            "Run `casestudies/dyn_sim/test_WT_FMU_drivetrain_sim.py` once to generate it."
        )

    wt = pd.read_csv(wt_csv)
    fmu = pd.read_csv(fmu_csv)

    # Use FMU time grid as the comparison baseline (typically coarser).
    t_fmu = np.asarray(fmu['t'], dtype=float)
    t_fmu = _ensure_increasing(t_fmu)
    if t_fmu.size == 0:
        raise ValueError("FMU CSV has empty time vector")

    wt_a = _align_df_on_time(wt, t_fmu, t_col='t')
    fmu_a = _align_df_on_time(fmu, t_fmu, t_col='t')

    # --- Speed bases ---
    # WT CSV: omega_*_pu are on omega_base_rpm stored in the CSV.
    wt_base_rpm = float(wt['omega_base_rpm'].iloc[0]) if 'omega_base_rpm' in wt.columns and len(wt) else np.nan
    # FMU CSV: rpm signals are converted to pu on omega_base_rpm stored in the CSV (from model parameter).
    fmu_base_rpm = float(fmu['omega_base_rpm'].iloc[0]) if 'omega_base_rpm' in fmu.columns and len(fmu) else np.nan

    # Fall back (older CSVs): infer from GenSpeed if omega_base_rpm not present.
    if (not np.isfinite(fmu_base_rpm)) or fmu_base_rpm <= 0.0:
        fmu_base_rpm = np.nan
        if 'GenSpeed' in fmu.columns:
            g = np.asarray(fmu['GenSpeed'], dtype=float)
            g = g[np.isfinite(g)]
            if g.size:
                fmu_base_rpm = float(np.median(g[: min(g.size, 200)]))
        if (not np.isfinite(fmu_base_rpm)) or fmu_base_rpm <= 0.0:
            fmu_base_rpm = 1.0

    # Derive comparable speed traces in RPM and pu-on-common-base.
    # Common base: use FMU base (so both can be compared directly as "pu (FMU base)").
    common_base_rpm = float(fmu_base_rpm)

    if 'omega_e_pu' in wt_a.columns and np.isfinite(wt_base_rpm) and wt_base_rpm > 0:
        wt_a['omega_e_rpm'] = wt_a['omega_e_pu'].to_numpy(dtype=float) * wt_base_rpm
        wt_a['omega_e_pu_common'] = wt_a['omega_e_rpm'].to_numpy(dtype=float) / common_base_rpm
    if 'omega_m_pu' in wt_a.columns and np.isfinite(wt_base_rpm) and wt_base_rpm > 0:
        wt_a['omega_m_rpm'] = wt_a['omega_m_pu'].to_numpy(dtype=float) * wt_base_rpm
        wt_a['omega_m_pu_common'] = wt_a['omega_m_rpm'].to_numpy(dtype=float) / common_base_rpm

    if 'GenSpeed' in fmu_a.columns:
        fmu_a['omega_e_rpm'] = fmu_a['GenSpeed'].to_numpy(dtype=float)
        fmu_a['omega_e_pu_common'] = _pu_from_rpm(fmu_a['omega_e_rpm'].to_numpy(dtype=float), common_base_rpm)
    if 'RotSpeed' in fmu_a.columns:
        fmu_a['omega_m_rpm'] = fmu_a['RotSpeed'].to_numpy(dtype=float)
        fmu_a['omega_m_pu_common'] = _pu_from_rpm(fmu_a['omega_m_rpm'].to_numpy(dtype=float), common_base_rpm)

    # --- Plots ---
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))
    fig.suptitle('WT model vs FMU drivetrain: comparable signals', fontsize=12)

    # 1) Power on system base
    if 'P_e_sys_pu' in wt_a.columns and 'P_e_sys_pu' in fmu_a.columns:
        axes[0].plot(t_fmu, wt_a['P_e_sys_pu'], label='WT: P_e', color=PLOT_COLORS[0], linewidth=1.4)
        axes[0].plot(t_fmu, fmu_a['P_e_sys_pu'], '--', label='FMU: P_e', color=PLOT_COLORS[1], linewidth=1.2)
    if 'P_ref_sys_pu' in wt_a.columns and 'P_ref_sys_pu' in fmu_a.columns:
        axes[0].plot(t_fmu, wt_a['P_ref_sys_pu'], ':', label='WT: P_ref', color=PLOT_COLORS[2], linewidth=1.2)
        axes[0].plot(t_fmu, fmu_a['P_ref_sys_pu'], '-.', label='FMU: P_ref', color=PLOT_COLORS[3], linewidth=1.2)
    axes[0].set_ylabel('Power (p.u.)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8)
    _plain_y(axes[0])

    # 2) Terminal voltage magnitude
    if 'v_bus_pu' in wt_a.columns and 'v_bus_pu' in fmu_a.columns:
        axes[1].plot(t_fmu, wt_a['v_bus_pu'], label='WT: |V_t|', color=PLOT_COLORS[0], linewidth=1.4)
        axes[1].plot(t_fmu, fmu_a['v_bus_pu'], '--', label='FMU: |V_t|', color=PLOT_COLORS[1], linewidth=1.2)
    axes[1].set_ylabel('|V_t| (p.u.)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8)
    _plain_y(axes[1])

    # 3) Speeds (pu on common base = FMU omega_m_rated)
    if 'omega_e_pu_common' in wt_a.columns:
        axes[2].plot(t_fmu, wt_a['omega_e_pu_common'], label='WT: ω_e', color=PLOT_COLORS[0], linewidth=1.4)
    if 'omega_m_pu_common' in wt_a.columns:
        axes[2].plot(t_fmu, wt_a['omega_m_pu_common'], ':', label='WT: ω_m', color=PLOT_COLORS[2], linewidth=1.2)
    if 'omega_e_pu_common' in fmu_a.columns:
        axes[2].plot(
            t_fmu,
            fmu_a['omega_e_pu_common'],
            '--',
            label='FMU: ω_e',
            color=PLOT_COLORS[1],
            linewidth=1.2,
        )
    if 'omega_m_pu_common' in fmu_a.columns:
        axes[2].plot(
            t_fmu,
            fmu_a['omega_m_pu_common'],
            '-.',
            label='FMU: ω_m',
            color=PLOT_COLORS[3],
            linewidth=1.2,
        )
    axes[2].set_ylabel('Speed (p.u.)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=8)
    _plain_y(axes[2])

    # 4) Pitch (deg)
    if 'pitch_deg' in wt_a.columns:
        axes[3].plot(t_fmu, wt_a['pitch_deg'], label='WT: pitch', color=PLOT_COLORS[0], linewidth=1.4)
    if 'BldPitch1' in fmu_a.columns:
        axes[3].plot(t_fmu, fmu_a['BldPitch1'], '--', label='FMU: pitch', color=PLOT_COLORS[1], linewidth=1.2)
    axes[3].set_ylabel('Pitch (deg)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='best', fontsize=8)
    _plain_y(axes[3])

    # 5) Wind (m/s)
    if 'wind_speed_mps' in wt_a.columns:
        axes[4].plot(t_fmu, wt_a['wind_speed_mps'], label='WT: wind', color=PLOT_COLORS[0], linewidth=1.4)
    # FMU has Wind1VelX and/or RtVAvgxh
    if 'Wind1VelX' in fmu_a.columns:
        axes[4].plot(t_fmu, fmu_a['Wind1VelX'], '--', label='FMU: Wind1VelX', color=PLOT_COLORS[1], linewidth=1.2)
    if 'RtVAvgxh' in fmu_a.columns:
        axes[4].plot(t_fmu, fmu_a['RtVAvgxh'], ':', label='FMU: RtVAvgxh', color=PLOT_COLORS[2], linewidth=1.2)
    axes[4].set_ylabel('Wind (m/s)')
    axes[4].set_xlabel('Time (s)')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='best', fontsize=8)
    _plain_y(axes[4])

    fig.tight_layout()

    plots_dir = os.path.join(project_root, 'casestudies', 'dyn_sim', 'logs', 'fmu_drivetrain', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_png = os.path.join(plots_dir, 'compare_WT_vs_FMU_results.png')
    fig.savefig(out_png, dpi=180)
    print(f"Saved comparison figure to {out_png}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()

