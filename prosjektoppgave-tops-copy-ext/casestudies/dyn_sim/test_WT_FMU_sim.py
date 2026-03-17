import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Like DynaWind: cwd = project root so extract(unzipdir='openfast_fmu') and wd path work
os.chdir(project_root)

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.dynamic as dps
import src.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':
    # Model loading and initialisation
    import casestudies.ps_data.test_WT_FMU_ as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)

    # UIC p_ref for power flow (FMU provides it during dynamics via connection)
    uic_model = ps.vsc['UIC_sig']
    uic_model.par['p_ref'][:] = 0.5
    uic_model.par['q_ref'][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()
    v0 = ps.v0.copy()

    fmu_models = [mdl for mdl in ps.dyn_mdls if hasattr(mdl, 'step_fmu')]
    gen_model = ps.gen['GEN']
    uic_name = uic_model.par['name'][0]
    gen_name = gen_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    t_end = 60
    dt = 0.01
    # Use dt=0.01 to match OpenFAST FMU (canHandleVariableCommunicationStepSize=false)
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=dt)

    # Runtime storage
    P_e_stored = []
    P_ref_stored = []
    GenSpeed_stored = []
    GenTq_stored = []
    v_bus = []

    # Initial point
    v0 = ps.solve_algebraic(0, x0)
    result_dict['Global', 't'].append(0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = ps.sys_data['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    gen_s_n = gen_model.par['S_n'][0]

    P_e_uic = uic_model.p_e(x0, v0)[0]
    P_ref_uic = uic_model.p_ref(x0, v0)[0]
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

    fmu_mdl = fmu_models[0] if fmu_models else None
    if fmu_mdl and hasattr(fmu_mdl, 'get_fmu_outputs'):
        GenTq, GenSpeed = fmu_mdl.get_fmu_outputs()
        GenSpeed_stored.append(GenSpeed)
        GenTq_stored.append(GenTq)

    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))

    # Simulation loop
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        for mdl in fmu_models:
            mdl.step_fmu(x, v, t, dt)

        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        P_e_uic = uic_model.p_e(x, v)[0]
        P_ref_uic = uic_model.p_ref(x, v)[0]
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

        if fmu_mdl and hasattr(fmu_mdl, 'get_fmu_outputs'):
            GenTq, GenSpeed = fmu_mdl.get_fmu_outputs()
            GenSpeed_stored.append(GenSpeed)
            GenTq_stored.append(GenTq)

        v_t_uic = uic_model.v_t(x, v)[0]
        v_bus.append(np.abs(v_t_uic))

    # Terminate FMU
    for mdl in fmu_models:
        if hasattr(mdl, 'terminate_fmu'):
            mdl.terminate_fmu()

    # Convert to DataFrame and plot
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    t_stored = result[('Global', 't')]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    fig.suptitle('test_WT_FMU: OpenFAST FMU co-simulation', fontsize=14)

    ax[0].plot(t_stored, P_e_stored, label='P_e (UIC electrical)', color='blue', linewidth=1.5)
    ax[0].plot(t_stored, P_ref_stored, '--', label='P_ref (FMU → UIC)', color='#FF1493', linewidth=1.5)
    ax[0].set_ylabel('Power (p.u., sys S_n)')
    ax[0].legend(loc='best')
    ax[0].grid(True, alpha=0.3)

    if GenSpeed_stored:
        ax[1].plot(t_stored, GenSpeed_stored, label='GenSpeed (FMU)', color='blue', linewidth=1.5)
        ax[1].set_ylabel('GenSpeed (rpm)')
        ax[1].legend(loc='best')
        ax[1].grid(True, alpha=0.3)

    if GenTq_stored:
        ax[2].plot(t_stored, GenTq_stored, label='GenTq (FMU)', color='#FF1493', linewidth=1.5)
        ax[2].set_ylabel('GenTq (kN-m)')
        ax[2].set_xlabel('Time (s)')
        ax[2].legend(loc='best')
        ax[2].grid(True, alpha=0.3)
    else:
        ax[2].plot(t_stored, v_bus, label='|v_bus|', color='blue', linewidth=1.5)
        ax[2].set_ylabel('Voltage (p.u.)')
        ax[2].set_xlabel('Time (s)')
        ax[2].legend(loc='best')
        ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
