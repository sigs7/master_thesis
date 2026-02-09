import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dyn_models.new_vsc import new_VSC
from src.solvers import EulerDAE

def test_new_vsc():
    """Test new_VSC using DAEModel framework with UIC-style control"""
    print("Testing new_VSC with UIC-style control...")
    print("=" * 50)
    
    # Create test parameters in the format expected by DAEModel
    test_params = {
        'name': ['VSC1'],
        'bus': [0],         # Bus index 
        'S_n': [50],        # 50 MVA
        'Ki': [0.01],       # Control gain 0.01 (reduced for stability)
        'Kv': [0.01],       # Voltage control gain 0.01 (reduced for stability)
        'xf': [0.1],        # Filter reactance 0.1
        'p_ref': [0.5],     # Power reference 
        'q_ref': [0.2],     # Reactive power reference 
    }
    
    # Create system parameters
    sys_params = {
        's_n': 100,        # 100 MVA base
        'f_n': 50,         # 50 Hz
        'bus_v_n': np.array([1.0]),  # 1.0 pu voltage
    }
    
    # Create VSC instance
    vsc = new_VSC(sys_par=sys_params, **test_params)
    
    # Initialize the model
    vsc.bus_idx['terminal'] = 0  # Connect to bus 0
    vsc.bus_idx_red['terminal'] = 0
    
    print("✓ VSC initialized successfully!")
    
    # Set up simulation
    t_end = 3.0
    dt = 0.01
    n_steps = int(t_end / dt)
    
    # Initialize state vector
    n_states = len(vsc.state_list())
    x = np.zeros(n_states)
    v = np.array([1.0 + 0.0j])  # Terminal voltage: 1.0 pu
    
    # Initialize the VSC
    vsc.init_from_load_flow(x, v, np.array([0.0 + 0.0j]))
    
    print(f"  - Initial vi_x: {x[0]:.4f} pu")
    print(f"  - Initial vi_y: {x[1]:.4f} pu")
    print(f"  - Initial angle: {x[2]:.4f} rad")
    
    # Set initial conditions: p_ref = 0.5, q_ref = 0.2, but start with non-zero internal voltage
    vsc._input_values['p_ref'] = 0.5  # Reference power is 0.5 pu
    vsc._input_values['q_ref'] = 0.2  # Reference reactive power is 0.2 pu
    
    # Set initial internal voltage to create initial current (error > 0 at t=0)
    x[0] = 1.05  # Start with vi_x = 1.05 pu (smaller initial error)
    x[1] = 0.0   # Start with vi_y = 0.0 pu
    x[2] = 0.0   # Start with angle = 0.0 rad (will be updated by init_from_load_flow)
    
    # Calculate initial current to verify
    initial_current = vsc.i_inj(x, v)
    initial_power = v[0] * np.conj(initial_current)
    print(f"  - Initial current: {complex(initial_current.item()):.4f} pu")
    print(f"  - Initial power: {complex(initial_power.item()):.4f} pu")
    
    # Create solver
    def state_derivatives(t, x, v):
        dx = np.zeros_like(x)
        vsc.state_derivatives(dx, x, v)
        return dx
    
    def solve_algebraic(t, x):
        return v  # Simple case: terminal voltage is constant
    
    solver = EulerDAE(state_derivatives, solve_algebraic, x0=x, t0=0.0, dt=dt, t_end=t_end)
    
    # Run simulation
    time = np.linspace(0, t_end, n_steps)
    vi_x_history = []
    vi_y_history = []
    angle_history = []
    current_history = []
    power_history = []
    error_history = []
    
    print("\nRunning simulation...")
    
    for i in range(n_steps):
        solver.step()
        t = solver.t
        
        # Get current state
        X = vsc.local_view(solver.x)
        vi_x = X['vi_x'][0]
        vi_y = X['vi_y'][0]
        angle = X['angle'][0]
        current = vsc.i_inj(solver.x, v)
        power = v[0] * np.conj(current)
        
        # Calculate control error
        s_ref = vsc._input_values['p_ref'] + 1j*vsc._input_values['q_ref']
        vi = vi_x + 1j*vi_y
        i_ref = np.conj(s_ref/vi)
        i_actual = current
        error = i_ref - i_actual
        
        # Store history
        vi_x_history.append(vi_x)
        vi_y_history.append(vi_y)
        angle_history.append(angle)
        current_history.append(current)
        power_history.append(power)
        error_history.append(error)
        
        if i % 100 == 0:  # Print every 1 second
            print(f"  t={t:.1f}s: vi_x={vi_x:.4f}, vi_y={vi_y:.4f}, current={complex(current.item()):.4f}, power={complex(power.item()):.4f}")
    
    # Plot 1: Original Analysis (Settling Time, Error, etc.)
    plt.figure(figsize=(15, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(time, vi_x_history, 'b-', linewidth=2, label='vi_x')
    plt.plot(time, vi_y_history, 'r-', linewidth=2, label='vi_y')
    plt.ylabel('Internal Voltage (pu)')
    plt.title(f'new_VSC Internal Voltage Components (UIC-style control, P_ref={vsc._input_values["p_ref"].item():.1f}, Q_ref={vsc._input_values["q_ref"].item():.1f}, Ki={vsc.par["Ki"][0]:.1f})')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time, [abs(c) for c in current_history], 'g-', linewidth=2, label='|Current|')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('Current Magnitude (pu)')
    plt.title('VSC Current Response')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time, [p.real for p in power_history], 'b-', linewidth=2, label='Active Power')
    plt.plot(time, [p.imag for p in power_history], 'r-', linewidth=2, label='Reactive Power')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.ylabel('Power (pu)')
    plt.title('VSC Power Response')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 1, 4)
    plt.plot(time, [abs(e) for e in error_history], 'm-', linewidth=2, label='|Current Error|')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Current Error (pu)')
    plt.title('VSC Control Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the first plot
    import os
    testing_folder = "testing"
    if not os.path.exists(testing_folder):
        os.makedirs(testing_folder)
    plot_filename_1 = os.path.join(testing_folder, "new_vsc_uic_response_p_and_q_ref.png")
    plt.savefig(plot_filename_1, dpi=300, bbox_inches='tight')
    print(f"Plot 1 saved to: {plot_filename_1}")
    plt.show()
    
    # Plot 2: New Feature Analysis (Angle and Grid-Forming Behavior)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, angle_history, 'm-', linewidth=2, label='dq-Frame Angle')
    plt.ylabel('Angle (rad)')
    plt.title('VSC dq-Frame Angle (Grid-Forming Behavior)')
    plt.legend()
    plt.grid(True)
    
    # Calculate frequency from angle derivative
    angle_derivative = np.gradient(angle_history, time)
    frequency = angle_derivative / (2 * np.pi)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, frequency, 'c-', linewidth=2, label='Instantaneous Frequency')
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Nominal 50 Hz')
    plt.ylabel('Frequency (Hz)')
    plt.title('VSC Frequency Generation')
    plt.legend()
    plt.grid(True)
    
    # Plot angle vs time with theoretical 50 Hz line
    theoretical_angle = 2 * np.pi * 50 * time
    
    plt.subplot(3, 1, 3)
    plt.plot(time, angle_history, 'm-', linewidth=2, label='Actual Angle')
    plt.plot(time, theoretical_angle, 'r--', linewidth=2, alpha=0.7, label='Theoretical 50 Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Angle Comparison: Actual vs Theoretical 50 Hz')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the second plot
    plot_filename_2 = os.path.join(testing_folder, "new_vsc_angle_analysis.png")
    plt.savefig(plot_filename_2, dpi=300, bbox_inches='tight')
    print(f"Plot 2 saved to: {plot_filename_2}")
    plt.show()
    
    
    # Calculate settling time based on current error (the regulated variable)
    tolerance = 0.02
    # Use the actual reference values from the VSC
    p_ref = vsc._input_values['p_ref']
    q_ref = vsc._input_values['q_ref']
    final_power = p_ref + 1j*q_ref  # Reference power from VSC
    
    print(f"\nSettling time analysis (based on current error):")
    print(f"  - Reference power: {complex(final_power.item()):.4f} pu")
    print(f"  - Final actual power: {complex(power_history[-1].item()):.4f} pu")
    print(f"  - Final current error: {abs(error_history[-1].item()):.4f} pu")
    print(f"  - Tolerance: {tolerance*100:.1f}%")
    
    settling_time = None
    min_error = float('inf')
    best_time = 0.0
    
    for i, (t, error) in enumerate(zip(time, error_history)):
        current_error_magnitude = abs(error)
        
        # Track the minimum error and corresponding time
        if current_error_magnitude < min_error:
            min_error = current_error_magnitude
            best_time = t
        
        # Check if within tolerance (current error should be small)
        if current_error_magnitude <= tolerance * 0.1:  # 5% of 0.1 pu reference
            settling_time = t
            break
    
    # Always report settling time
    if settling_time is not None:
        print(f"\n✓ Settling time ({tolerance*100:.1f}% tolerance): {settling_time:.2f} seconds")
    else:
        print(f"\n⚠ System did not settle within {t_end} seconds")
        print(f"   Best convergence at t={best_time:.2f}s with error={min_error.item():.4f} pu")
        print(f"   Final error: {abs(power_history[-1] - final_power).item():.4f} pu")
    
    # Final values
    final_current = current_history[-1]
    final_power = power_history[-1]
    final_vi_x = vi_x_history[-1]
    final_vi_y = vi_y_history[-1]
    
    print(f"\nFinal Results:")
    print(f"  - Final current: {complex(final_current.item()):.4f} pu")
    print(f"  - Final power: {complex(final_power.item()):.4f} pu")
    print(f"  - Final vi_x: {final_vi_x:.4f} pu")
    print(f"  - Final vi_y: {final_vi_y:.4f} pu")
    
    print(f"\n✓ new_VSC test completed successfully!")
    return True

if __name__ == "__main__":
    test_new_vsc()
