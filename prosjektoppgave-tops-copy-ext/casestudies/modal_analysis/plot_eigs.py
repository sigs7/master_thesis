import src.dynamic as dps
import src.modal_analysis as dps_mdl
import src.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import casestudies.ps_data.k2a_regulated as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em', 'non_conj'], damp_threshold=0.3)
    rev = ps_lin.rev
    labels = ps.gen['GEN'].par['name']
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)
        mode_shape_max = ms[np.argmax(np.abs(ms))]
        print(abs(ms) / np.abs(mode_shape_max))

    # Colourmap is [red, blue, green, purple, ...]
    plt.show(block=True)
