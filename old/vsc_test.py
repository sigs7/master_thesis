import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dyn_models.new_vsc import new_VSC
from src.solvers import EulerDAE

def test_vsc():
    """Test VSC using DAEModel framework with proper solver"""
    print("Testing VSC with DAEModel framework...")
    print("=" * 50)
    
    # Create test parameters in the format expected by DAEModel
    test_params = {
        'name': ['VSC1'],
        'bus': ['B1'],
        'S_n': [50],       # 50 MVA
        'K_p': [0.0],      # Proportional gain
        'K_i': [1.8],      # Integral gain  ## 
        'xf': [0.1],       # Reactance
        'i_ref': [0.0],    # Current reference 
    }
    
    # Create system parameters
    sys_params = {
        's_n': 100,        # System base power (MVA)
        'bus_v_n': np.array([20.0]),  # Bus voltage rating
    }
    
    try:
        # Create VSC - pass parameters as keyword arguments
        vsc = new_VSC(sys_par=sys_params, **test_params)
        print("✓ VSC created successfully!")
        print(f"  - Number of units: {vsc.n_units}")
        print(f"  - Number of states: {vsc.n_states}")
        print(f"  - PI controller K_p: {vsc.par['K_p'][0]}")
        print(f"  - PI controller K_i: {vsc.par['K_i'][0]}")
        
        # Create state vector
        x = np.zeros(vsc.n_states * vsc.n_units)
        v = np.array([1.0 + 0.0j])
        
        # Initialize VSC
        vsc.init_from_load_flow(x, v, None)
        print("✓ VSC initialized successfully!")
        
        # Set initial conditions: i_ref = 0, but start with non-zero current (e > 0 at t=0)
        vsc._input_values['i_ref'] = 0.0  # Reference is 0
        
        # Set initial integral state to create initial error
        X0 = vsc.local_view(x)
        X0['x_pi'][:] = -0.05  # Start with non-zero integral to create initial current
        
        # Set initial current to non-zero to create initial error
        vsc._last_current = 0.5  # Start with 0.5 pu current (error = 0 - 0.5 = -0.5)
        
        # Calculate initial current to verify
        initial_current = vsc.i_inj(x, v)
        
        print(f"  - Initial current: {float(initial_current):.4f} pu")
        print(f"  - Initial error: {float(vsc._input_values['i_ref'] - initial_current):.4f} pu")
        
        # Run simulation
        t_end = 3.0
        dt = 0.01
        
        # Create solver functions
        def f(t, x, v):
            """State derivatives function"""
            dx = np.zeros_like(x)
            vsc.state_derivatives(dx, x, v)
            
            # Debug: print state derivatives every 100 steps
            if int(t / dt) % 100 == 0:
                X = vsc.local_view(x)
                current_debug = vsc.i_inj(x, v)
                print(f"    Debug t={t:.2f}: current={float(current_debug):.4f}, integral={float(X['x_pi'][0]):.4f}")
                print(f"    Debug t={t:.2f}: dx_integral={float(dx[0]):.6f}")
                print(f"    Debug t={t:.2f}: i_ref={float(vsc._input_values['i_ref']):.4f}, error={float(vsc._input_values['i_ref'] - current_debug):.4f}")
            
            return dx
        
        def g_inv(t, x):
            """Algebraic equations solver - just return constant voltage"""
            return v
        
        solver = EulerDAE(f, g_inv, t0=0.0, x0=x, t_end=t_end, dt=dt)
        
        print("✓ Solver created and setup successfully!")
        n_steps = int(t_end / dt)
        
        time = np.linspace(0, t_end, n_steps)
        current_history = []
        error_history = []
        integral_history = []
        
        print("\nRunning simulation...")
        
        for i, t in enumerate(time):
            # Step the solver first
            solver.step()
            
            # Get current state after stepping - use solver's state vector
            X = vsc.local_view(solver.x)
            current = vsc.i_inj(solver.x, v)
            error = vsc._input_values['i_ref'] - current
            integral = X['x_pi'][0]
            
            # Store history
            current_history.append(current)
            error_history.append(error)
            integral_history.append(integral)
            
            if i % 100 == 0:  # Print every 1 second
                print(f"  t={t:.1f}s: current={float(current):.4f}, error={float(error):.4f}, integral={float(integral):.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(time, current_history, 'b-', linewidth=2, label='Current')
        plt.axhline(y=0, color='r', linestyle='--', label='Reference (0)')
        plt.ylabel('Current (pu)')
        plt.title('VSC Current Response (i_ref=0, e>0 at t=0)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time, error_history, 'r-', linewidth=2, label='Error')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('Error (pu)')
        plt.title('VSC Error Response')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time, integral_history, 'g-', linewidth=2, label='Integral')
        plt.xlabel('Time (s)')
        plt.ylabel('Integral (pu)')
        plt.title('PI Controller Integral Term')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot in a testing folder
        import os
        testing_folder = "testing"
        if not os.path.exists(testing_folder):
            os.makedirs(testing_folder)
        
        plot_filename = os.path.join(testing_folder, "vsc_current_respons_i_ref_0.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
        
        plt.show()
        
        # Calculate settling time (2% tolerance)
        tolerance = 0.02
        final_value = 0.0  # Reference is 0
        
        settling_time = None
        for i, (t, current) in enumerate(zip(time, current_history)):
            if abs(current - final_value) <= tolerance * abs(final_value) + tolerance:
                settling_time = t
                break
        
        if settling_time is not None:
            print(f"\n✓ Settling time (2% tolerance): {settling_time:.3f} seconds")
        else:
            print(f"\n✗ System did not settle within tolerance")
        
        
        print("✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_vsc()
