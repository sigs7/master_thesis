import sys
import os
# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.dynamic as dps
import src.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # region Model loading and initialisation stage
    import casestudies.ps_data.test_WT as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    uic_model = ps.vsc['UIC_sig']
    wt_model = ps.windturbine['WindTurbine']
    
    # Get wind turbine parameters
    R = wt_model.par['R'][0]  # Rotor radius in m
    omega_m_rated = wt_model.par['omega_m_rated'][0]  # Rated speed in RPM
    omega_m_rated_rad_s = omega_m_rated * 2 * np.pi / 60  # Convert to rad/s
    wind_rated = wt_model.par['wind_rated'][0]  # Rated wind speed in m/s
    
    # Calculate optimal TSR from rated conditions
    optimal_tsr = 1 * R / wind_rated
    """ 
    # Load wind data file to get initial wind speed
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    wind_file_path = os.path.join(project_root, 'wind_data', '10mps_NTM_3xDTU10MW_IECKAI_VS_T1.hh')
    
    # Load wind data (skip first line, read first two columns)
    wind_data = np.loadtxt(wind_file_path, skiprows=1, usecols=(0, 1))
    wind_times = wind_data[:, 0]  # First column: time in seconds
    wind_speeds = wind_data[:, 1]  # Second column: wind speed in m/s
    
    # Use first wind speed value from file as initial wind speed
    init_wind_speed = wind_speeds[0]  # m/s
     """
    # Calculate initial rotor speed from initial wind speed
    init_wind_speed = 8.0  # m/s
    init_omega_m_rad_s = optimal_tsr * init_wind_speed / R
    
    # Load MPT table and interpolate to get optimal power for initial rotor speed
    from scipy.interpolate import interp1d
    mpt_file_path = os.path.join(project_root, 'wind_data', 'MPT.csv')
    mpt_data = np.loadtxt(mpt_file_path, delimiter='\t', skiprows=2)
    rotor_speed_RPM = mpt_data[:, 0]  # Rotor speeds in RPM
    optimal_powers = mpt_data[:, 1]  # Optimal powers in pu (WT base)
    
    # Create interpolator for MPT curve (using RPM directly)
    mpt_interp = interp1d(rotor_speed_RPM, optimal_powers, kind='linear',
                          bounds_error=False, fill_value=(0.0, optimal_powers[-1]))
    
    # Convert initial rotor speed from rad/s to RPM for interpolation
    init_omega_m_RPM = init_omega_m_rad_s * 60 / (2 * np.pi)
    
    # Get initial P_ref from MPT curve (in WT local base, pu)
    P_rated_sys = wt_model.par['P_rated'][0]  # Rated power in pu (system base)
    P_mpt_init = float(mpt_interp(init_omega_m_RPM))
    
    # Convert P_rated from system base to WT base for comparison
    sys_s_n = ps.s_n  # System base MVA
    wt_s_n = wt_model.par['S_n'][0]  # WT S_n
    P_rated_wt = P_rated_sys * sys_s_n / wt_s_n  # Convert to WT base
    
    P_ref_wt_pu = min(P_mpt_init, P_rated_wt)  # Limit to rated power (both in WT base)
    
    # Convert P_ref from WT base to UIC base
    uic_s_n = uic_model.par['S_n'][0]  # UIC S_n
    init_P_ref = P_ref_wt_pu * wt_s_n / uic_s_n  # Convert to UIC base
    
    uic_model.par['p_ref'][:] = init_P_ref
    uic_model.par['q_ref'][:] = 0.0
    
    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    wt_model = ps.windturbine['WindTurbine']
    uic_model = ps.vsc['UIC_sig']
    gen_model = ps.gen['GEN']  # Infinite bus generator
    
    # Override q_ref input value to ensure it's 0 (init_from_load_flow sets it from load flow solution)
    uic_model._input_values['q_ref'][:] = 0.0
    wt_name = wt_model.par['name'][0]
    uic_name = uic_model.par['name'][0]
    gen_name = gen_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    t_end = 120 # Simulation time

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
    # endregion

    # region Print initial conditions
    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = np.angle(ps.v_0)  # In radians
    print(f'Voltages (pu): {v_bus_mag}')
    print(f'Voltage angles: {v_bus_angle} \n')
    print(f'state description: \n {ps.state_desc} \n')
    print(f'Initial values on all state variables (WT and UIC) : \n {x0} \n')
    # endregion

    # region Runtime variables
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    P_ref_stored = []
    v_bus = []
    omega_m_hist = []
    omega_e_hist = []
    pitch_angle_hist = []
    wind_speed_hist = []
    i_a_mag_hist = []
    i_a_angle_hist = []
    P_gen_stored = []  # Infinite bus power injection
    Q_gen_stored = []  # Infinite bus reactive power
    Q_e_uic_stored = []  # UIC reactive power (actual)
    Q_ref_uic_stored = []  # UIC reactive power (reference)

    # endregion

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]

        # Short circuit
        """ if 1 <= t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e5
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0 """

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables

        # Store bus voltage (at UIC terminal) - use UIC model's v_t() method for consistency
        # Note: Voltages are in system base per-unit (voltage base, NOT power base)
        # They do NOT depend on UIC's S_n - voltages are always on the bus voltage base
        v_t_uic = uic_model.v_t(x, v)[0]  # Terminal voltage from UIC model
        v_bus.append(np.abs(v_t_uic))  # Stores magnitude
        # Convert powers to system base for consistent plotting
        P_m_local = wt_model.P_m(x, v)[0]  # In WT local base
        P_e_uic = wt_model.P_e(x, v)[0]  # From UIC.p_e() = s_e.real, where s_e = v_t*conj(i_a)
        # s_e is in UIC local base: v_t (system base voltage) * i_a (UIC local base current) = UIC local base power
        P_ref_uic = wt_model.P_ref(x, v)[0]  # Returns UIC local base (WT local → UIC local conversion)
        # Get bases for conversions
        sys_s_n = wt_model.sys_par['s_n']
        uic_s_n = uic_model.par['S_n'][0]
        wt_s_n = wt_model.par['S_n'][0]
        gen_s_n = gen_model.par['S_n'][0]
        # Convert to system base
        P_m_stored.append(P_m_local * wt_s_n / sys_s_n)  # WT local → system
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)  # UIC local → system
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)  # UIC local → system
        # Infinite bus power injection (in generator local base, convert to system base)
        P_gen_local = gen_model.p_e(x, v)[0]  # Generator local base
        Q_gen_local = gen_model.q_e(x, v)[0]  # Generator local base
        P_gen_stored.append(P_gen_local * gen_s_n / sys_s_n)  # Generator local → system
        Q_gen_stored.append(Q_gen_local * gen_s_n / sys_s_n)  # Generator local → system
        # UIC reactive power (in UIC local base, convert to system base)
        # Calculate reactive power at internal voltage (excluding reactive power consumed by xf)
        # Use i_a (current through xf) to calculate power at internal voltage
        X = uic_model.local_view(x)
        vi = X['vi_x'][0] + 1j*X['vi_y'][0]  # Internal voltage
        i_a = uic_model.i_a(x, v)[0]  # Current through xf (from terminal to internal)
        s_internal = vi * np.conj(i_a)  # Power at internal voltage
        Q_e_uic_local = s_internal.imag  # Reactive power at internal voltage (UIC local base)
        Q_ref_uic_local = uic_model.q_ref(x, v)[0]  # UIC local base
        Q_e_uic_stored.append(Q_e_uic_local * uic_s_n / sys_s_n)  # UIC local → system
        Q_ref_uic_stored.append(Q_ref_uic_local * uic_s_n / sys_s_n)  # UIC local → system
        wt_states = wt_model.local_view(x)
        omega_m_hist.append(wt_states['omega_m'][0])
        omega_e_hist.append(wt_states['omega_e'][0])
        # Pitch angle is stored as state variable
        pitch_angle_val = wt_states['pitch_angle'][0] if 'pitch_angle' in wt_states.dtype.names else 0.0
        pitch_angle_hist.append(float(pitch_angle_val * 180 / np.pi))  # Convert to degrees and ensure scalar
        # Wind speed
        wind_speed_hist.append(wt_model.wind_speed(x, v))
        # UIC armature current
        i_a = uic_model.i_a(x, v)[0]
        i_a_mag_hist.append(np.abs(i_a))
        i_a_angle_hist.append(np.angle(i_a) * 180 / np.pi)  # Convert to degrees
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    t_stored = result[('Global', 't')]

    # First figure: Wind Turbine.
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    fig1.suptitle('Wind Turbine', fontsize=14)

    # Rotational and electrical speeds
    ax1[0].plot(t_stored, omega_m_hist, label='ω_m (mechanical speed)', color='blue', linewidth=1.5)
    ax1[0].plot(t_stored, omega_e_hist, label='ω_e (electrical speed)', color='#FF1493', linewidth=1.5)  # deeppink
    ax1[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Rated speed (1.0 p.u.)')
    ax1[0].set_ylabel('Speed (p.u., base = ω_m_rated)')
    ax1[0].legend(loc='best')
    ax1[0].grid(True, alpha=0.3)

    # Pitch angle
    ax1[1].plot(t_stored, pitch_angle_hist, label='Pitch angle', color='blue', linewidth=1.5)
    ax1[1].set_ylabel('Pitch angle (deg)')
    ax1[1].legend(loc='best')
    ax1[1].grid(True, alpha=0.3)

    # Wind speed
    ax1[2].plot(t_stored, wind_speed_hist, label='Wind speed', color='#FF1493', linewidth=1.5)  # deeppink
    ax1[2].set_ylabel('Wind speed (m/s)')
    ax1[2].set_xlabel('Time (s)')
    ax1[2].legend(loc='best')
    ax1[2].grid(True, alpha=0.3)

    # Second figure: UIC
    fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
    fig2.suptitle('UIC', fontsize=14)

    # Voltages (magnitude)
    # Note: All voltages are in system base per-unit (voltage base, NOT power base)
    vi_x = result[(uic_name, 'vi_x')]
    vi_y = result[(uic_name, 'vi_y')]
    vi_mag = np.sqrt(vi_x**2 + vi_y**2)
    ax2[0].plot(t_stored, vi_mag, label='|v_i| (internal voltage)', color='#FF1493', linewidth=1.5)  # deeppink
    ax2[0].plot(t_stored, np.array(v_bus), label='|v_bus| (terminal voltage)', color='blue', linewidth=1.5)
    ax2[0].set_ylabel('Voltage (p.u., system voltage base)')
    ax2[0].legend(loc='best')
    ax2[0].grid(True, alpha=0.3)

    # Internal voltage components
    ax2[1].plot(t_stored, vi_x, label='v_i_x (real)', color='#FF1493', linewidth=1.5)  # deeppink
    ax2[1].plot(t_stored, vi_y, label='v_i_y (imaginary)', color='blue', linewidth=1.5)
    ax2[1].set_ylabel('Internal voltage (p.u.)')
    ax2[1].legend(loc='best')
    ax2[1].grid(True, alpha=0.3)
    ax2[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Current magnitude
    ax2[2].plot(t_stored, i_a_mag_hist, label='|i_a| (armature current)', color='#FF1493', linewidth=1.5)  # deeppink
    ax2[2].set_ylabel('Current magnitude (p.u., UIC base)')
    ax2[2].legend(loc='best')
    ax2[2].grid(True, alpha=0.3)

    # Current angle
    ax2[3].plot(t_stored, i_a_angle_hist, label='∠i_a (armature current angle)', color='blue', linewidth=1.5)
    ax2[3].set_ylabel('Current angle (deg)')
    ax2[3].set_xlabel('Time (s)')
    ax2[3].legend(loc='best')
    ax2[3].grid(True, alpha=0.3)

    # Third figure: Power
    fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    fig3.suptitle('Power', fontsize=14)

    # Power comparison (P_m, P_e, P_ref)
    ax3[0].plot(t_stored, P_m_stored, label='P_m (mechanical)', color='orange', linewidth=1.5)
    ax3[0].plot(t_stored, P_e_stored, label='P_e (electrical)', color='#FF1493', linewidth=1.5)  # deeppink
    ax3[0].plot(t_stored, P_ref_stored, label='P_ref (reference)', color='blue', linewidth=1.5, linestyle='--')
    ax3[0].set_ylabel('Power (p.u., system base)')
    ax3[0].legend(loc='best')
    ax3[0].grid(True, alpha=0.3)

    # Infinite bus power (P and Q in same subplot)
    ax3[1].plot(t_stored, P_gen_stored, label='P_inf (infinite bus)', color='blue', linewidth=1.5)
    ax3[1].plot(t_stored, Q_gen_stored, label='Q_inf (infinite bus)', color='#FF1493', linewidth=1.5)  # deeppink
    ax3[1].set_ylabel('Power (p.u., system base)')
    ax3[1].legend(loc='best')
    ax3[1].grid(True, alpha=0.3)

    # UIC reactive power (actual and reference)
    ax3[2].plot(t_stored, Q_e_uic_stored, label='Q_e (UIC actual)', color='#FF1493', linewidth=1.5)  # deeppink
    ax3[2].plot(t_stored, Q_ref_uic_stored, label='Q_ref (UIC reference)', color='blue', linewidth=1.5, linestyle='--')
    ax3[2].set_ylabel('Reactive Power (p.u., system base)')
    ax3[2].set_xlabel('Time (s)')
    ax3[2].legend(loc='best')
    ax3[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
    # endregion
