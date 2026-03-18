import matplotlib.pyplot as plt
import control as ctrl
from matplotlib.patches import Arc
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from ipywidgets import interact, FloatSlider, widgets
from IPython.display import display

"""
------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------  Coordinate Notebook Utils  --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
"""

# Function to change from abc to alpha-beta coordinates
def abc_to_alphabeta(abc: np.ndarray) -> np.ndarray:
    """ Change from abc to alpha-beta coordinates. This is done by the rotational matrix shown below. 
     The variable "abc" is a numpy array with the three phase values, i.e. a vector [Ia, Ib, Ic] or [Va, Vb, Vc]. """
    T_alpha_beta = (2/3) * np.array([
        [1, -0.5, -0.5],
        [0, np.sqrt(3)/2, -np.sqrt(3)/2]
    ])
    return T_alpha_beta @ abc # Matrix multiplication for the Clarke transformation matrix

def alphabeta_to_dq(alpha_beta : np.ndarray, theta : float) -> np.ndarray:
    """ Change from alpha-beta to dq coordinates. This is done by the rotational matrix shown below. 
     The variable "alpha_beta" is a numpy array with the two phase values, i.e. a vector [I_alpha, I_beta] or [V_alpha, V_beta]. 
     The variable "theta" is the angle of rotation. """
    T_dq = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return T_dq @ alpha_beta # Matrix multiplication for the Park transformation matrix

# Plotting
def plot_Coordinate_Display(t, Vm, Im, Iphase_diff):
    """
    This function plots the voltages and currents in the abc, alpha-beta, and dq coordinate systems.

    Has a time slider to display time-dependency.

    Args:
        t (_type_): _description_
        Vm (_type_): _description_
        Im (_type_): _description_
        Iphase_diff (_type_): _description_
    """
    wt = 2 * np.pi * 50 * t
    theta = wt

    # ABC phase voltages at given time
    Va = Vm * np.cos(wt)
    Vb = Vm * np.cos(wt - 2 * np.pi / 3)
    Vc = Vm * np.cos(wt + 2 * np.pi / 3)
    Vabc = np.array([[Va], [Vb], [Vc]])

    # ABC phase currents at given time; -20 degrees compared to the voltages and 0.8 magnitude
    phase_diff = -20 * np.pi/180
    Ia = Im * np.cos(wt + Iphase_diff)
    Ib = Im * np.cos(wt - 2 * np.pi / 3 + Iphase_diff)
    Ic = Im * np.cos(wt + 2 * np.pi / 3 + Iphase_diff)
    Iabc = np.array([[Ia],[Ib],[Ic]])

    V_alpha_beta = abc_to_alphabeta(Vabc)
    I_alpha_beta = abc_to_alphabeta(Iabc)
    Vdq = alphabeta_to_dq(V_alpha_beta, theta)
    Idq = alphabeta_to_dq(I_alpha_beta, theta)
    fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Add d and q axis vectors
    ax1.quiver(0, 0, 1.35*np.cos(wt), 1.35*np.sin(wt), angles='xy', scale_units='xy', scale=1, color='grey', width=0.003)
    ax1.quiver(0, 0, -1.35*np.sin(wt), 1.35*np.cos(wt), angles='xy', scale_units='xy', scale=1, color='grey', width=0.003)

    # Annotate d and q axes
    ax1.text(1.35*np.cos(wt) + 0.15, 1.35*np.sin(wt), 'd', fontsize=12, color='grey', ha='center')
    ax1.text(-1.35*np.sin(wt) + 0.15, 1.35*np.cos(wt), 'q', fontsize=12, color='grey', ha='center')

    # Plot alpha-beta voltages and currents
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.quiver(0, 0, V_alpha_beta[0], V_alpha_beta[1], angles='xy', scale_units='xy', scale=1, color='forestgreen', label=r'$V_{\alpha\beta}$', width=0.005)
    ax1.quiver(0, 0, I_alpha_beta[0], I_alpha_beta[1], angles='xy', scale_units='xy', scale=1, color='cyan', label=r'$I_{\alpha\beta}$', width=0.005)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()
    ax1.set_title(r'$\alpha \beta$ Voltages and Currents')
    ax1.grid(True)

    # Calculate and annotate the angle of the voltage vector
    angle = np.arctan2(V_alpha_beta[1], V_alpha_beta[0])
    angle_deg = np.rad2deg(angle).item()  # Convert to scalar

    # Annotate the angle at the base of the voltage vector
    ax1.text(0.2, 0.2, f'{angle_deg:.2f}°', fontsize=12, color='black')


    # Draw an arc to represent the angle
    arc = Arc((0, 0), 0.5, 0.5, angle=0, theta1=0, theta2=angle_deg, color='black', linestyle='--')
    ax1.add_patch(arc)

    # Annotate axis as alpha and beta
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\beta$')

    # Plot abc voltages and currents
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.quiver(0, 0, np.cos(wt), np.sin(wt), angles='xy', scale_units='xy', scale=1, color='g', label='Va', width=0.005)
    ax2.quiver(0, 0, np.cos(wt + 4 * np.pi / 3), np.sin(wt + 4 * np.pi / 3), angles='xy', scale_units='xy', scale=1, color='r', label='Vb', width=0.005)
    ax2.quiver(0, 0, np.cos(wt + 2 * np.pi / 3), np.sin(wt + 2 * np.pi / 3), angles='xy', scale_units='xy', scale=1, color='b', label='Vc', width=0.005)
    ax2.quiver(0, 0, Ia, .8 * np.sin(wt + phase_diff), angles='xy', scale_units='xy', scale=1, color='cyan', label='Ia', width=0.005)
    ax2.quiver(0, 0, Ib, .8 * np.sin(wt - 2 * np.pi / 3 + phase_diff), angles='xy', scale_units='xy', scale=1, color='magenta', label='Ib', width=0.005)
    ax2.quiver(0, 0, Ic, .8 * np.sin(wt + 2 * np.pi / 3 + phase_diff), angles='xy', scale_units='xy', scale=1, color='yellow', label='Ic', width=0.005)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    ax2.set_title(r'$abc$ Voltages and Currents')
    ax2.grid(True)

    # Plot dq voltages and currents
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.quiver(0, 0, Vdq[0, 0], Vdq[1, 0], angles='xy', scale_units='xy', scale=1, color='forestgreen', label=r'$V_{dq}$', width=0.005)
    ax3.quiver(0, 0, Idq[0, 0], Idq[1, 0], angles='xy', scale_units='xy', scale=1, color='cyan', label=r'$I_{dq}$', width=0.005)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()
    ax3.set_title(r'$dq$ Coordinates')
    ax3.set_xlabel('d')
    ax3.set_ylabel('q')
    ax3.grid(True)

    plt.suptitle(r'$abc$ and $\alpha \beta$ Voltages and Currents at $t = {:.4f}$'.format(t), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



"""
------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------  Phasor Diagram Utils  ----------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
"""



def plot_phasor_diagram(phasors, xlim : tuple, ylim : tuple):
    """
    Plots the phasor diagram for the given complex numbers.

    Parameters:
    phases (dict): A dictionary containing the complex numbers for the phasors. 
                   Expected keys are 'Vs', 'Vt', 'Vg', 'Eq', 'Eq_t', 'I', 'Id', 'Iq'.

    Returns:
    None
    """
    plt.figure()
    # Define a color map for the phasors
    colors = {
        'Vs': 'dodgerblue',
        'Vt': 'deepskyblue',
        'Vg': 'steelblue',
        'Eq': 'green',
        'Eq_t': 'limegreen',
        'I': 'red',
        'Id': 'orangered',
        'Iq': 'darkred'
    }
    
    # Plot all vectors in the dictionary with colors
    for key, value in phasors.items():
        color = colors.get(key, 'black')  # Default to black if key is not in colors
        plt.quiver(0, 0, np.real(value), np.imag(value), angles='xy', scale_units='xy', scale=1, color=color, label=key, width=0.0025)
        plt.text(np.real(value), np.imag(value), key, fontsize=12, color=color)

    # Calculate delta and phi if Eq and I are provided
    Eq = phasors.get('Eq', None)
    I = phasors.get('I', None)
    delta = np.angle(Eq) if Eq is not None else None
    phi = -np.angle(I) if I is not None else None


    # Add angle arcs if delta and phi are calculated
    if delta is not None:
        arc_delta = Arc([0, 0], 0.5, 0.5, theta1=0, theta2=np.degrees(delta), color='blue', linestyle='--')
        plt.gca().add_patch(arc_delta)
        plt.text(0.25, 0.05, r'$\delta$', color='blue', fontsize=12)

    if phi is not None:
        arc_phi = Arc([0, 0], 0.3, 0.3, theta1=-np.degrees(phi), theta2=0, color='magenta', linestyle='--')
        plt.gca().add_patch(arc_phi)
        plt.text(0.15, -0.05, r'$\phi$', color='magenta', fontsize=12)

    # Add labels and legend
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Phasor Diagram')
    plt.tight_layout()
    plt.show()

# Define a function to update the phasor diagram
def update_phasor_diagram(P, Q, Vs, Xd, Xd_t, Xq, Xt, Xl):
    # Recalculate the variables based on the slider values
    I = (P - 1j*Q) / Vs
    phi = np.abs(np.angle(I))
    Vt = Vs + I * 1j*(Xl)
    Vg = Vs + I * 1j*(Xt + Xl)
    xq = Xq + Xt + Xl
    xd = Xd + Xt + Xl
    xd_t = Xd_t + Xt + Xl
    EQ = Vs + I*1j*(xq)
    delta = np.angle(EQ)
    beta = delta + phi
    Id_m = np.abs(I)*np.sin(beta)
    Iq_m = np.abs(I)*np.cos(beta)
    Id = Id_m * np.exp(1j*(delta - np.pi/2))
    Iq = Iq_m * np.exp(1j*(delta))
    Eq = Vg + Id*1j*(Xd) + Iq*1j*Xq
    Eq_t = Vs + Id*1j*(xd_t) + Iq*1j*xq



    phasors = {
        'Vs': Vs,
        'Vt': Vt,
        'Vg': Vg,
        'Eq': Eq,
        'Eq_t': Eq_t,
        'I': I,
        'Id': Id,
        'Iq': Iq
    }

    xlim = (-max(np.real(value) for value in phasors.values())*0.2, max(np.real(value) for value in phasors.values())*1.3)
    ylim = (min(np.imag(value) for value in phasors.values())*1.2,max(np.imag(value) for value in phasors.values())*1.2)    

    plot_phasor_diagram(phasors, ylim=ylim, xlim=xlim)






"""
------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------  Small distrubance utils  --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
"""

def getPhasorsDictionary_SmallDist(P, Q, Vs, Xd, Xd_t, Xq, Xt, Xl):
    # Recalculate the variables based on the slider values
    I = (P - 1j*Q) / Vs
    phi = np.abs(np.angle(I))
    Vt = Vs + I * 1j*(Xl)
    Vg = Vs + I * 1j*(Xt + Xl)
    xq = Xq + Xt + Xl
    xd = Xd + Xt + Xl
    xd_t = Xd_t + Xt + Xl
    EQ = Vs + I*1j*(xq)
    delta = np.angle(EQ)
    beta = delta + phi
    Id_m = np.abs(I)*np.sin(beta)
    Iq_m = np.abs(I)*np.cos(beta)
    Id = Id_m * np.exp(1j*(delta - np.pi/2))
    Iq = Iq_m * np.exp(1j*(delta))
    Eq = Vg + Id*1j*(Xd) + Iq*1j*Xq
    Eq_t = Vs + Id*1j*(xd_t) + Iq*1j*xq

    phasors = {
        'Vs': Vs,
        'Vt': Vt,
        'Vg': Vg,
        'Eq': Eq,
        'Eq_t': Eq_t,
        'I': I,
        'Id': Id,
        'Iq': Iq
    }
    return phasors


# Define the function to calculate eigenvalues
def calculate_eigenvalues(H, D, K_E_t, f_N):
    A = np.array([[0, 2 * np.pi * f_N],
                  [-K_E_t / (2 * H), -D / (2 * H)]])
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues

# Define the function to plot the root locus
def plot_root_locus(H, D, K_E_t, f_N):
    eigenvalues = calculate_eigenvalues(H, D, K_E_t, f_N)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='red')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.xlim(-4, 1)
    plt.ylim(-15, 15)
    plt.title('Root Locus')
    plt.grid(True)
    
    # Annotate damping and frequency
    for eig in eigenvalues:
        damping = -eig.real / np.sqrt(eig.real**2 + eig.imag**2) * 100
        frequency = np.abs(eig.imag) / (2 * np.pi)
        plt.annotate(f'{damping:.2f} %, {frequency:.2f} Hz', (eig.real, eig.imag), textcoords="offset points", xytext=(10,-10), ha='center')
    # Mark the area 0.1-2 Hz and below 10% damping indicating electromechanical modes
    freq_range = np.linspace(0.1, 2, 1000)
    damping_ratio = 0.1
    real_part = -damping_ratio * 2 * np.pi * freq_range
    imag_part = 2 * np.pi * freq_range * np.sqrt(1 - damping_ratio**2)
    plt.fill_betweenx(imag_part, real_part, 0, color='yellow', alpha=0.3, label='Electromechanical Modes (0.1-2 Hz, <10% damping)')
    plt.fill_betweenx(-imag_part, real_part, 0, color='yellow', alpha=0.3)
    plt.legend()
    
    plt.show()


'''
------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------  Large disturbance utils  --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
'''
# Define the power curve function
def power_curve(delta, P_m, E, Vs, x_dt):
    return np.sin(delta) * (abs(E) * abs(Vs)) / x_dt

# Define the integrate power curve function
def integrate_power_curve(delta_start, delta_end, P_m, E, Vs, x_dt):
    result, _ = quad(power_curve, delta_start, delta_end, args=(P_m, E, Vs, x_dt))
    return result

# Function to find delta_0 where power curve intersects with P_m
def find_delta_0(P_m, E, Vs, x_dt):
    func = lambda delta: power_curve(delta, P_m, E, Vs, x_dt) - P_m
    delta_0 = fsolve(func, 0)[0]
    return delta_0

# Function to find delta_end where power curve intersects with P_m after delta_clear
def find_delta_end(delta_clear, P_m, E, Vs, x_dt):
    func = lambda delta: power_curve(delta, P_m, E, Vs, x_dt) - P_m
    delta_end = fsolve(func, np.pi)[0]  # Start the search from delta_clear + 0.1
    return delta_end

# Function to update the plot
def update_plot(delta_clear, P_m, P_e_sc, E, Vs, x_dt):
    delta_0 = find_delta_0(P_m, E, Vs, x_dt)
    delta_end = find_delta_end(delta_clear, P_m, E, Vs, x_dt)
    delta = np.linspace(0, np.pi, 1000)
    power = power_curve(delta, P_m, E, Vs, x_dt)
    
    A_1 = (P_m-P_e_sc) * (delta_clear - delta_0)
    A_2 = integrate_power_curve(delta_clear, delta_end, P_m, E, Vs, x_dt) - P_m * (delta_end - delta_clear)
    
    plt.figure(figsize=(10, 6))
    plt.plot(delta, power, label='Power Curve')
    plt.axhline(P_m, color='gray', linestyle='--', label='P_m')
    plt.axhline(P_e_sc, color='green', linestyle='--', label='P_e_sc')

    
    # Fill Area 1
    plt.fill_between(delta, P_e_sc, P_m, where=(delta >= delta_0) & (delta <= delta_clear), color='blue', alpha=0.3, label='Area 1')
    
    # Fill Area 2
    plt.fill_between(delta, P_m, power, where=(delta > delta_clear) & (delta <= delta_end), color='green', alpha=0.3, label='Area 2')
    
    plt.axvline(delta_clear, color='red', linestyle='--', label='Delta Clear')
    plt.axvline(delta_end, color='orange', linestyle='--', label='Delta End')
    plt.xlabel('Delta [rad]')
    plt.ylabel('Power')
    plt.legend()
    plt.title(f'Power Curve with Areas A_1 and A_2\nA_1: {A_1:.2f}, A_2: {A_2:.2f}')
    plt.show()


'''
------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------  VOLTAGE STABILITY UTILS  --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
'''

def calculate_open_loop_transfer_function(H, D, w_N, K_E_t):
    return ctrl.TransferFunction([1, 0], [2*H, D, w_N*K_E_t], name='Open Loop')

def calculate_governor_transfer_function(Tw, Kgov):
    return Kgov * ctrl.TransferFunction([-Tw, 1], [0.5*Tw, 1], name='GOV')

def calculate_combined_transfer_function(H_OL, GOV):
    return ctrl.TransferFunction(H_OL * GOV, name='Open Loop with Governor')

def plot_bode_response(H_OL, H_OLwithGOV):
    plt.figure(figsize=(12, 6))
    ctrl.bode_plot(H_OL, dB=True, label='Open Loop', display_margins=False)
    ctrl.bode_plot(H_OLwithGOV, dB=True, display_margins=True)
    plt.legend(['Open Loop', 'Open Loop with Governor'])
    plt.show()

def update_plots_small_signal(Tw, Kgov, H, D, w_N, K_E_t):
    H_OL = calculate_open_loop_transfer_function(H, D, w_N, K_E_t)
    GOV = calculate_governor_transfer_function(Tw, Kgov)
    H_OLwithGOV = calculate_combined_transfer_function(H_OL, GOV)
    
    plot_bode_response(H_OL, H_OLwithGOV)

def calc_damping_and_freq(eigenvalue):
    damping = -eigenvalue.real / np.sqrt(eigenvalue.real**2 + eigenvalue.imag**2) * 100
    frequency = np.abs(eigenvalue.imag) / (2 * np.pi)
    print(f'Damping of eigenvalue {eigenvalue} is: {damping:.2f} % and frequency: {frequency:.2f} Hz')
    return damping, frequency

# NOTEBOOK 7

# Define the function to update the transfer functions and plots
def update_transfer_functions_AVR(k_p, k_i, k_d, T_E, K_e, K_3, T_d0_t, K_6, T_m):
    # AVR block:

    AVR_PID = ctrl.tf( ctrl.tf([k_p], [1])+ctrl.tf([k_i], [1, 0])+ctrl.tf([k_d, 0], [1]) , name = r"G_{avr}(s)")

    #Exciter block:
    # Exciter time constant, assume not tunable, T_e = 0.1 TOPS standard value
    Gexc = ctrl.tf([K_e], [T_E, 1], name = r"G_{exc}(s)")

    # Generator block:
    Ggen = ctrl.tf([K_3], [K_3*T_d0_t, 1], name = r"G_{gen}(s)")

    # Feedback Block 
    Hmeas = ctrl.tf([1], [T_m, 1], name = r"H_{meas}(s)")

    G = ctrl.tf(Gexc*Ggen*AVR_PID*K_6, name = r"G(s)")
    L = ctrl.tf(G*Hmeas, name = r"L(s)")
    CL = ctrl.tf(G/(1+L), name = r"CL(s)")

    return L,CL

def poles_interactive_plot_AVR(CL):

    poles = ctrl.poles(CL)
    plt.figure()
    plt.scatter(poles.real, poles.imag, marker='x', color='red')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Poles of the Closed-Loop Transfer Function')
    plt.grid(True)

    # Annotate damping and frequency
    for eig in poles:
        if abs(eig) < 1e-10:
            continue
        damping = -eig.real/abs(eig)
        if damping < 0.99:
            frequency = np.abs(eig.imag) / (2 * np.pi)
            plt.annotate(f'{damping*100:.2f} %, {frequency:.2f} Hz', (eig.real, eig.imag), textcoords="offset points", xytext=(10, -10), ha='center')

    # Mark the area 0.1-2 Hz and below 10% damping indicating electromechanical modes
    freq_range = np.linspace(0.1, 2, 1000)
    damping_ratio = 0.1
    real_part = -damping_ratio * 2 * np.pi * freq_range
    imag_part = 2 * np.pi * freq_range * np.sqrt(1 - damping_ratio**2)
    plt.fill_betweenx(imag_part, real_part, 0, color='yellow', alpha=0.3, label='Electromechanical Modes (0.1-2 Hz, <10% damping)')
    plt.fill_betweenx(-imag_part, real_part, 0, color='yellow', alpha=0.3)
    plt.legend(loc='upper left')

    # plt.show()

def bode_interactive_plot_AVR(CL):

    response = ctrl.frequency_response(CL)
    ctrl.bode_plot(response, initial_phase=0, display_margins=True, dB=True)
    # plt.show()

def stepresponse_interactive_plot_AVR(CL):


    t, y = ctrl.step_response(CL)
    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.title('Step Response of the Closed-Loop Transfer Function')
    # plt.show()

def interactive_AVR(plot_type, kp, ki, kd, K_e, T_E, K_3, T_d0_t, K_6, T_m):
    L, CL = update_transfer_functions_AVR(kp, ki, kd, T_E, K_e, K_3, T_d0_t, K_6, T_m)
    if plot_type == "POLES":
        poles_interactive_plot_AVR(CL)
    elif plot_type == "BODE":
        bode_interactive_plot_AVR(CL)
    elif plot_type == "STEP":
        stepresponse_interactive_plot_AVR(CL)
    # plt.show()


def display_interactive_AVR():
    ui = widgets.HBox([
        widgets.VBox([
            widgets.Dropdown(options=["POLES", "BODE", "STEP"], description='Plot Type:'),
            widgets.FloatSlider(value=10.0, min=0.0, max=100.0, step=0.05, description='kp'),
            widgets.FloatSlider(value=2.0, min=0.0, max=100.0, step=0.05, description='ki'),
            widgets.FloatSlider(value=3.0, min=0.0, max=100.0, step=0.05, description='kd'),
        ]),
        widgets.VBox([
            widgets.FloatText(value=1.0, description='K_e'),
            widgets.FloatText(value=0.1, description='T_E'),
            widgets.FloatText(value=0.3, description='K_3'),
            widgets.FloatText(value=8.0, description='T_d0_t'),
            widgets.FloatText(value=0.4, description='K_6'),
            widgets.FloatText(value=0.2, description='T_m'),
        ])
    ])
    out = widgets.interactive_output(interactive_AVR, {
        'plot_type': ui.children[0].children[0],
        'kp': ui.children[0].children[1],
        'ki': ui.children[0].children[2],
        'kd': ui.children[0].children[3],
        'K_e': ui.children[1].children[0],
        'T_E': ui.children[1].children[1],
        'K_3': ui.children[1].children[2],
        'T_d0_t': ui.children[1].children[3],
        'K_6': ui.children[1].children[4],
        'T_m': ui.children[1].children[5],
    })

    #Error here; Display function is a Notebook function i.e. this only works when called in a jupyter notebook
    display(ui, out)


def load_model_NO_AVR():
    return {
        'base_mva': 100, #'base_mva': system base/nominal complex power in [MVA].
        'f': 50, #'f': system frequency in [Hz].
        'slack_bus': 'B3', #'slack_bus': reference busbar with zero phase angle.

        'buses': [
            ['name',    'V_n'],
            ['B1',         10],
            ['B2',        132],
            ['B3',        132],
        ],
        #'V_n': base/nominal voltage in [kV].

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L2-3_1',        'B2',     'B3',        250,      100,    132,    'PF',   0,    0.2,     0],
            ['L2-3_2',        'B2',     'B3',        250,      110,    132,    'PF',   0,    0.2,     0],
        ],


        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['T1',         'B1',     'B2',    100,         10,      132,   0, 0.1],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',     100,     10,     80,  1.1314,      5,      0,     1.0,   0.66,    0.3,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
                ['IB',      'B3',    1e8,    132,    -80,  1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
            ],
        },

    }

def load_model_wAVR(K, T_a, T_b, T_e, E_min, E_max):
    return {
        'base_mva': 100, #'base_mva': system base/nominal complex power in [MVA].
        'f': 50, #'f': system frequency in [Hz].
        'slack_bus': 'B3', #'slack_bus': reference busbar with zero phase angle.

        'buses': [
            ['name',    'V_n'],
            ['B1',         10],
            ['B2',        132],
            ['B3',        132],
        ],
        #'V_n': base/nominal voltage in [kV].

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L2-3_1',        'B2',     'B3',        250,      100,    132,    'PF',   0,    0.2,     0],
            ['L2-3_2',        'B2',     'B3',        250,      110,    132,    'PF',   0,    0.2,     0],
        ],


        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['T1',         'B1',     'B2',    100,         10,      132,   0, 0.1],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',     100,     10,     80,  1.1314,      5,      0,     1.0,   0.66,    0.3,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
                ['IB',      'B3',    1e8,    132,    -80,  1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
            ],
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', K, T_a, T_b, T_e, E_min, E_max],
            ]
        },

    }




'''
------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------  FREQUENCY STABILITY UTILS  --------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
'''



def update_transfer_functions_GOV(k_p, k_i, k_d, rho, T_w, K_1, D, H, w_N):
    #PID block:
    G_pid = ctrl.tf( ctrl.tf([k_p], [1])+ctrl.tf([k_i], [1, 0])+ctrl.tf([k_d, 0], [1]) , name = r"G_{pid}(s)")
    # Turbine block:
    G_t = ctrl.tf([-T_w,1], [0.5*T_w, 1], name = r"G_{t}(s)")
    # GOV block:
    G_GOV = ctrl.tf(G_t*G_pid/(1+rho*G_t+G_pid), name = r"G_{GOV}(s)")
    # GEN block:
    G_gen = ctrl.tf([1,0], [2*H, D, K_1*w_N], name = r"G_{gen}(s)")
    # Open loop:
    L = ctrl.tf(G_GOV*G_gen, name = r"L(s)")
    # Closed loop:
    CL = ctrl.tf(L/(1+L), name = r"CL(s)")
    return L, CL

def poles_interactive_plot_GOV(CL):

    poles = ctrl.poles(CL)
    plt.figure()
    plt.scatter(poles.real, poles.imag, marker='x', color='red')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Poles of the Closed-Loop Transfer Function')
    plt.grid(True)

    # Annotate damping and frequency
    min_damping = float('inf')
    min_damping_pole = None
    for eig in poles:
        if abs(eig) < 1e-10:
            continue
        damping = -eig.real/abs(eig)
        if damping < min_damping:
            min_damping = damping
            min_damping_pole = eig

    if min_damping_pole is not None and min_damping < 0.99:
        frequency = np.abs(min_damping_pole.imag) / (2 * np.pi)
        plt.annotate(f'{min_damping*100:.2f} %, {frequency:.2f} Hz', (min_damping_pole.real, min_damping_pole.imag), textcoords="offset points", xytext=(10, -10), ha='center')

    # Mark the area 0.1-2 Hz and below 10% damping indicating electromechanical modes
    freq_range = np.linspace(0.1, 2, 1000)
    damping_ratio = 0.1
    real_part = -damping_ratio * 2 * np.pi * freq_range
    imag_part = 2 * np.pi * freq_range * np.sqrt(1 - damping_ratio**2)
    plt.fill_betweenx(imag_part, real_part, 0, color='yellow', alpha=0.3, label='Electromechanical Modes (0.1-2 Hz, <10% damping)')
    plt.fill_betweenx(-imag_part, real_part, 0, color='yellow', alpha=0.3)
    plt.legend(loc='upper left')

    # plt.show()

def bode_interactive_plot_GOV(CL):
    response = ctrl.frequency_response(CL)
    ctrl.bode_plot(response, initial_phase=0, display_margins=True, dB=True)
    # plt.show()

def stepresponse_interactive_plot_GOV(CL):
    t, y = ctrl.step_response(CL)
    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    plt.title('Step Response of the Closed-Loop Transfer Function')
    # plt.show()

def interactive_GOV(plot_type, k_p, k_i, k_d, rho, T_w, K_1, D, H, w_N):
    L, CL = update_transfer_functions_GOV(k_p, k_i, k_d, rho, T_w, K_1, D, H, w_N)
    if plot_type == "POLES":
        poles_interactive_plot_GOV(CL)
    elif plot_type == "BODE":
        bode_interactive_plot_GOV(CL)
    elif plot_type == "STEP":
        stepresponse_interactive_plot_GOV(CL)
    plt.show()

def display_interactive_GOV():
    ui = widgets.HBox([
        widgets.VBox([
            widgets.Dropdown(options=["POLES", "BODE", "STEP"], description='Plot Type:'),
            widgets.FloatSlider(value=2.5, min=0.0, max=100.0, step=0.05, description='kp'),
            widgets.FloatSlider(value=0.5, min=0.0, max=100.0, step=0.05, description='ki'),
            widgets.FloatSlider(value=0, min=0.0, max=100.0, step=0.05, description='kd'),
            widgets.FloatSlider(value=0.06, min=0.0, max=1.0, step=0.01, description='rho')
        ]),
        widgets.VBox([
            widgets.FloatText(value=1, description='T_w'),
            widgets.FloatText(value=1.75, description='K_1'),
            widgets.FloatText(value=10, description='D'),
            widgets.FloatText(value=5, description='H')
        ])
    ])

    out = widgets.interactive_output(interactive_GOV, {
        'plot_type': ui.children[0].children[0],
        'k_p': ui.children[0].children[1],
        'k_i': ui.children[0].children[2],
        'k_d': ui.children[0].children[3],
        'rho': ui.children[0].children[4],
        'T_w': ui.children[1].children[0],
        'K_1': ui.children[1].children[1],
        'D': ui.children[1].children[2],
        'H': ui.children[1].children[3],
        'w_N': widgets.fixed(2 * np.pi * 50)
    })

    #Error here; Display function is a Notebook function i.e. this only works when called in a jupyter notebook
    display(ui, out)


#  'gov': {
#      'TGOV1': [
#          ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
#          ['GOV1', 'G1', 0.05, 0.02, 0, 1, 0.1, 0.09, 0.2],
#      ]
#  },

def load_model_NO_GOV():
    return {
        'base_mva': 100, #'base_mva': system base/nominal complex power in [MVA].
        'f': 50, #'f': system frequency in [Hz].
        'slack_bus': 'B3', #'slack_bus': reference busbar with zero phase angle.

        'buses': [
            ['name',    'V_n'],
            ['B1',         10],
            ['B2',        132],
            ['B3',        132],
        ],
        #'V_n': base/nominal voltage in [kV].

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L2-3_1',        'B2',     'B3',        250,      100,    132,    'PF',   0,    0.2,     0],
            ['L2-3_2',        'B2',     'B3',        250,      110,    132,    'PF',   0,    0.2,     0],
        ],


        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['T1',         'B1',     'B2',    100,         10,      132,   0, 0.1],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',     100,     10,     80,  1.1314,      5,      0,     1.0,   0.66,    0.3,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
                ['IB',      'B3',    1e8,    132,    -80,  1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
            ],
        },

    }

def load_model_wGovernor(R, D_t, V_min, V_max, T_1, T_2, T_3):
    return {
        'base_mva': 100, #'base_mva': system base/nominal complex power in [MVA].
        'f': 50, #'f': system frequency in [Hz].
        'slack_bus': 'B3', #'slack_bus': reference busbar with zero phase angle.

        'buses': [
            ['name',    'V_n'],
            ['B1',         10],
            ['B2',        132],
            ['B3',        132],
        ],
        #'V_n': base/nominal voltage in [kV].

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L2-3_1',        'B2',     'B3',        250,      100,    132,    'PF',   0,    0.2,     0],
            ['L2-3_2',        'B2',     'B3',        250,      110,    132,    'PF',   0,    0.2,     0],
        ],


        'transformers': [
            ['name', 'from_bus', 'to_bus', 'S_n', 'V_n_from', 'V_n_to', 'R', 'X'],
            ['T1',         'B1',     'B2',    100,         10,      132,   0, 0.1],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',     100,     10,     80,  1.1314,      5,      0,     1.0,   0.66,    0.3,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
                ['IB',      'B3',    1e8,    132,    -80,  1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       0.254,      0.273,       2.49,      10000,         0.06,       0.15],
            ],
        },

        'gov': {
            'TGOV1': [
                ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
                ['GOV1', 'G1', R, D_t, V_min, V_max, T_1, T_2, T_3],
            ]
        },

    }