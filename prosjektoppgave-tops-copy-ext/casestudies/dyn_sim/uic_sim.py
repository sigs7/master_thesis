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

    # region Model loading and initialisation
    import casestudies.ps_data.test_WT as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    # Access the UIC_sig_pq VSC defined in the test_WT model
    WT = ps.windturbine['WindTurbine']
    wind_speed = 8.0
    ideal_tsr = 9.0
    start_rotor_speed_rad_s = ideal_tsr * wind_speed / WT.par['R'][0]
    start_rotor_speed_RPM = start_rotor_speed_rad_s * 60 / (2 * np.pi)
    WT._load_MPT_table()
    P_ref = WT._mpt_interp(start_rotor_speed_rad_s) * WT.par['S_n'][0] / ps.vsc['UIC_sig_pq'].par['S_n'][0]

    ps.vsc['UIC_sig_pq'].par['p_ref'][:] = P_ref 

    """ WT = ps.windturbine['WindTurbine']
    UIC = ps.vsc['UIC_sig']
    radius = WT.par['R'][0]
    ideal_tsr = 9.0 
    start_wind_speed = 8  # m/s
    start_rotor_speed_rad_s = ideal_tsr * start_wind_speed / radius  # rad/s (TSR = omega*R/v)
    WT._load_MPT_table()  # Load MPT table before use (_mpt_interp is created lazily)
    reference_active_power = WT._mpt_interp(start_rotor_speed_rad_s) * WT.par['S_n'][0] / UIC.par['S_n'][0]
    UIC.par['p_ref'][:] = reference_active_power
    UIC.par['q_ref'][:] = 0.0 """
    
    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    wt_model = ps.windturbine['WindTurbine']
    uic_model = ps.vsc['UIC_sig_pq']
    gen_model = ps.gen['GEN']  # Infinite bus generator
    
    # Override q_ref input value to ensure it's 0 (init_from_load_flow sets it from load flow solution)
    #uic_model._input_values['q_ref'][:] = 0.0
    wt_name = wt_model.par['name'][0]
    uic_name = uic_model.par['name'][0]
    gen_name = gen_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    t_end = 30 # Simulation time

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
    # endregion

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = np.angle(ps.v_0)  # In radians
    print(f'Voltages (pu): {v_bus_mag}')
    print(f'Voltage angles: {v_bus_angle} \n')
    print(f'state description: \n {ps.state_desc} \n')
    print(f'Initial values on all state variables (WT and UIC) : \n {x0} \n')
    

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
    P_gen_stored = [] 
    Q_gen_stored = [] 
    # Bus-side UIC power (actual and reference)
    P_uic_bus_actual = []
    Q_uic_bus_actual = []
    P_uic_bus_ref = []
    Q_uic_bus_ref = []

    # endregion

    # Store initial point (t0=0, x0, v0) so plots include first time step.
    # Use the same algebraic solution as the DAE solver (solve_algebraic(0,x0)), not power-flow voltage.
    # Otherwise the first point uses PF v while the solver uses Y*v=i_inj(x0) v, causing a visible jump.
    v0 = ps.solve_algebraic(0, x0)
    result_dict['Global', 't'].append(0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = wt_model.sys_par['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    wt_s_n = wt_model.par['S_n'][0]
    gen_s_n = gen_model.par['S_n'][0]
    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))
    P_m_local = wt_model.P_m(x0, v0)[0]
    P_e_uic = wt_model.P_e(x0, v0)[0]
    P_ref_uic = wt_model.P_ref(x0, v0)[0]
    P_m_stored.append(P_m_local * wt_s_n / sys_s_n)
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)
    P_gen_local = gen_model.p_e(x0, v0)[0]
    Q_gen_local = gen_model.q_e(x0, v0)[0]
    P_gen_stored.append(P_gen_local * gen_s_n / sys_s_n)
    Q_gen_stored.append(Q_gen_local * gen_s_n / sys_s_n)

    # UIC bus-side actual and reference at t0
    X = uic_model.local_view(x0)
    vi = X['vi_x'][0] + 1j*X['vi_y'][0]
    i_a = uic_model.i_a(x0, v0)[0]
    s_bus_actual = uic_model.s_e(x0, v0)[0]  # bus-side S
    # Internal reference S at vi
    s_ref_internal = uic_model.p_ref(x0, v0)[0] + 1j * uic_model.q_ref(x0, v0)[0]
    # Transform internal reference to bus-side: S_ext = S_int - j*xf*|I_a|^2
    xf = uic_model.par['xf'][0]
    s_bus_ref = s_ref_internal - 1j * xf * (np.abs(i_a) ** 2)

    P_uic_bus_actual.append(s_bus_actual.real * uic_s_n / sys_s_n)
    Q_uic_bus_actual.append(s_bus_actual.imag * uic_s_n / sys_s_n)
    P_uic_bus_ref.append(s_bus_ref.real * uic_s_n / sys_s_n)
    Q_uic_bus_ref.append(s_bus_ref.imag * uic_s_n / sys_s_n)
    wt_states = wt_model.local_view(x0)
    omega_m_hist.append(wt_states['omega_m'][0])
    omega_e_hist.append(wt_states['omega_e'][0])
    pitch_angle_val = wt_states['pitch_angle'][0] if 'pitch_angle' in wt_states.dtype.names else 0.0
    pitch_angle_hist.append(float(pitch_angle_val * 180 / np.pi))
    wind_speed_hist.append(wt_model.wind_speed(x0, v0))
    i_a_mag_hist.append(np.abs(i_a))
    i_a_angle_hist.append(np.angle(i_a) * 180 / np.pi)

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        sc_bus_idx = ps.vsc['UIC_sig_pq'].bus_idx_red['terminal'][0]

        # Short circuit
        """ if 1 <= t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e5
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0 """

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables

        v_t_uic = uic_model.v_t(x, v)[0] 
        v_bus.append(np.abs(v_t_uic))  
        P_m_local = wt_model.P_m(x, v)[0] 
        P_e_uic = wt_model.P_e(x, v)[0]  
        P_ref_uic = wt_model.P_ref(x, v)[0]  
        sys_s_n = wt_model.sys_par['s_n']
        uic_s_n = uic_model.par['S_n'][0]
        wt_s_n = wt_model.par['S_n'][0]
        gen_s_n = gen_model.par['S_n'][0]
        P_m_stored.append(P_m_local * wt_s_n / sys_s_n)  # WT local → system
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)  
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)  
        P_gen_local = gen_model.p_e(x, v)[0]  
        Q_gen_local = gen_model.q_e(x, v)[0]  
        P_gen_stored.append(P_gen_local * gen_s_n / sys_s_n)  
        Q_gen_stored.append(Q_gen_local * gen_s_n / sys_s_n)  

        # UIC bus-side actual and reference
        X = uic_model.local_view(x)
        vi = X['vi_x'][0] + 1j*X['vi_y'][0]  # Internal voltage
        i_a = uic_model.i_a(x, v)[0]  # Current through xf (from terminal to internal)
        s_bus_actual = uic_model.s_e(x, v)[0]
        s_ref_internal = uic_model.p_ref(x, v)[0] + 1j * uic_model.q_ref(x, v)[0]
        xf = uic_model.par['xf'][0]
        s_bus_ref = s_ref_internal - 1j * xf * (np.abs(i_a) ** 2)

        P_uic_bus_actual.append(s_bus_actual.real * uic_s_n / sys_s_n)
        Q_uic_bus_actual.append(s_bus_actual.imag * uic_s_n / sys_s_n)
        P_uic_bus_ref.append(s_bus_ref.real * uic_s_n / sys_s_n)
        Q_uic_bus_ref.append(s_bus_ref.imag * uic_s_n / sys_s_n)
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
    # All plot series have same length: index 0 = initial point (t0, x0, v0), then one per solver step
    n_pts = len(t_stored)
    assert n_pts == len(Q_uic_bus_actual) == len(P_uic_bus_actual), "plot series length mismatch"

    # First figure: Wind Turbine.
    fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    fig1.suptitle('Wind Turbine', fontsize=14)

    # Rotational and electrical speeds
    ax1[0].plot(t_stored, omega_m_hist, label='ω_m (mechanical speed)', color='blue', linewidth=1.5)
    ax1[0].plot(t_stored, omega_e_hist, label='ω_e (electrical speed)', color='#FF1493', linewidth=1.5)  # deeppink
    ax1[0].set_ylabel('Speed (p.u., base ω_m_rated)')
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
    ax2[0].set_ylabel('Voltage (p.u., sys V_n)')
    ax2[0].legend(loc='best')
    ax2[0].grid(True, alpha=0.3)

    # Internal voltage components
    ax2[1].plot(t_stored, vi_x, label='v_i_x (real)', color='#FF1493', linewidth=1.5)  # deeppink
    ax2[1].plot(t_stored, vi_y, label='v_i_y (imaginary)', color='blue', linewidth=1.5)
    ax2[1].set_ylabel('Internal voltage (p.u., sys V_n)')
    ax2[1].legend(loc='best')
    ax2[1].grid(True, alpha=0.3)
    ax2[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Current magnitude
    ax2[2].plot(t_stored, i_a_mag_hist, label='|i_a| (armature current)', color='#FF1493', linewidth=1.5)  # deeppink
    ax2[2].set_ylabel('Current (p.u., UIC S_n)')
    ax2[2].legend(loc='best')
    ax2[2].grid(True, alpha=0.3)

    # Current angle
    ax2[3].plot(t_stored, i_a_angle_hist, label='∠i_a (armature current angle)', color='blue', linewidth=1.5)
    ax2[3].set_ylabel('Current angle (deg)')
    ax2[3].set_xlabel('Time (s)')
    ax2[3].legend(loc='best')
    ax2[3].grid(True, alpha=0.3)

    # Third figure: Power
    # 0: WT powers, 1: UIC P at bus, 2: UIC Q at bus, 3: Infinite bus P/Q
    fig3, ax3 = plt.subplots(4, 1, sharex=True, figsize=(9, 12))
    fig3.suptitle('Power', fontsize=14)

    # Power comparison (WT view): P_m, P_e at UIC, and P_ref from WT
    ax3[0].plot(t_stored, P_m_stored, label='P_m (mechanical, WT)', color='orange', linewidth=1.5)
    ax3[0].plot(t_stored, P_e_stored, label='P_e (electrical at UIC)', color='#FF1493', linewidth=1.5)  # deeppink
    ax3[0].plot(t_stored, P_ref_stored, label='P_ref, WT → UIC (command)', color='blue', linewidth=1.5, linestyle='--')
    ax3[0].set_ylabel('Power (p.u., sys S_n)')
    ax3[0].legend(loc='best')
    ax3[0].grid(True, alpha=0.3)

    # UIC bus-side active power: actual vs reference (including xf losses)
    ax3[1].plot(t_stored, P_uic_bus_actual, label='P_UIC actual at bus', color='#FF1493', linewidth=1.5)
    ax3[1].plot(t_stored, P_uic_bus_ref, '--', label='P_UIC ref at bus (from internal ref)', color='blue', linewidth=1.5)
    ax3[1].set_ylabel('P at bus (p.u., sys S_n)')
    ax3[1].legend(loc='best')
    ax3[1].grid(True, alpha=0.3)

    # UIC bus-side reactive power: actual vs external reference (constant Q_ref)
    # External Q_ref is the case-data setpoint on UIC base, converted to system base
    sys_s_n_plot = wt_model.sys_par['s_n']
    uic_s_n_plot = uic_model.par['S_n'][0]
    q_ref_ext_sys = uic_model.par['q_ref'][0] * uic_s_n_plot / sys_s_n_plot
    ax3[2].plot(t_stored, Q_uic_bus_actual, label='Q_UIC actual at bus', color='#FF1493', linewidth=1.5)
    ax3[2].axhline(y=q_ref_ext_sys, linestyle='--', color='blue',
                   label='Q_UIC ref at bus (external setpoint, const.)')
    ax3[2].set_ylabel('Q at bus (p.u., sys S_n)')
    ax3[2].legend(loc='best')
    ax3[2].grid(True, alpha=0.3)

    # Infinite bus power (P and Q in same subplot)
    ax3[3].plot(t_stored, P_gen_stored, label='P_inf (infinite bus)', color='blue', linewidth=1.5)
    ax3[3].plot(t_stored, Q_gen_stored, label='Q_inf (infinite bus)', color='#FF1493', linewidth=1.5)  # deeppink
    ax3[3].set_ylabel('P, Q_inf (p.u., sys S_n)')
    ax3[3].set_xlabel('Time (s)')
    ax3[3].legend(loc='best')
    ax3[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)
    # endregion
