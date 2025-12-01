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

    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    wt_model = ps.windturbine['WindTurbine']
    uic_model = ps.vsc['UIC_sig']
    wt_name = wt_model.par['name'][0]
    uic_name = uic_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    t_end = 100 # Simulation time

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

    # event_flag1 = True  # Create a different flag for each system event
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

        v_bus.append(np.abs(v[0]))  # Stores magnitude of generator terminal voltage
        # Convert powers to system base for consistent plotting
        P_m_local = wt_model.P_m(x, v)[0]  # In WT local base
        P_e_uic = wt_model.P_e(x, v)[0]  # From UIC.p_e() = s_e.real, where s_e = v_t*conj(i_a)
        # s_e is in UIC local base: v_t (system base voltage) * i_a (UIC local base current) = UIC local base power
        P_ref_sys = wt_model.P_ref(x, v)[0]  # Returns system base (WT local → system conversion)
        # Get bases for conversions
        sys_s_n = wt_model.sys_par['s_n']
        uic_s_n = uic_model.par['S_n'][0]
        # Convert to system base
        P_m_stored.append(P_m_local * wt_model._local_to_sys[0])  # WT local → system
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)  # UIC local → system
        P_ref_stored.append(P_ref_sys)  # Already in system base
        wt_states = wt_model.local_view(x)
        omega_m_hist.append(wt_states['omega_m'][0])
        omega_e_hist.append(wt_states['omega_e'][0])
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    t_stored = result[('Global', 't')]

    fig, ax = plt.subplots(3, sharex=True, figsize=(9, 8))
    fig.suptitle('UIC internal voltage and wind turbine response')

    vi_x = result[(uic_name, 'vi_x')]
    vi_y = result[(uic_name, 'vi_y')]
    vi_mag = np.sqrt(vi_x**2 + vi_y**2)
    ax[0].plot(t_stored, vi_mag, label='|v_i|')
    ax[0].plot(t_stored, np.array(v_bus), label='|v_bus|')
    ax[0].set_ylabel('Voltage (p.u.)')
    ax[0].legend()

    ax[1].plot(t_stored, omega_m_hist, label='ω_m')
    ax[1].plot(t_stored, omega_e_hist, label='ω_e')
    ax[1].set_ylabel('Speed (p.u.)')
    ax[1].legend()

    ax[2].plot(t_stored, P_m_stored, label='P_m (mech)')
    ax[2].plot(t_stored, P_e_stored, label='P_e (elec)')
    ax[2].plot(t_stored, P_ref_stored, label='P_ref')
    ax[2].set_ylabel('Power (p.u., system base)')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()

    plt.show(block = True)
    # endregion
