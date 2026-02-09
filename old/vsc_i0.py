"""  
Goal: Model a vsc without PLL

1. Create a DC source with current control -> i_ref = 0
2. Validate time constants: T = K_i/X
3. Implement current control i_ref not 0

- Use existing solvers
- Use existing blocks
- Differential equations are divided into steps (functions with different objectives)

"""

from .blocks import *

class new_VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}
    
    def state_list(self):
        """Return list of states for this VSC"""
        return ['x_pi']  # Only the PI integral state is needed

    def input_list(self):
        """Input list for VSC"""
        return ['i_ref']

    def state_derivatives(self, dx, x, v):
        """Calculate state derivatives (PI controller action)"""
        dX = self.local_view(dx)
        X = self.local_view(x)
        
        # Get current reference
        i_ref = self._input_values['i_ref']
        
        # Calculate current from PI controller output
        current = self.i_inj(x, v)
        
        # Update the last current for next iteration
        self._last_current = current
        
        # Calculate error
        error = i_ref - current
        
        # PI controller: integral term derivative
        dX['x_pi'][:] = error

    def i_inj(self, x, v):
        """Calculate current from PI controller output: i = vi / xf"""
        X = self.local_view(x)
        
        # Get current reference
        i_ref = self._input_values['i_ref']
        
        # Calculate current directly from PI controller output
        # PI controller output: vi = e * K_i / s
        K_p = self.par['K_p'][0]
        K_i = self.par['K_i'][0]
        
        # For the first call, use the stored last current to calculate error
        if not hasattr(self, '_last_current'):
            self._last_current = 0.0
        
        # Calculate error using last current value
        error_P = i_ref - self._last_current
         
        # PI controller output
        vi = K_p * error_P + K_i * X['x_pi']
        
        # Current is directly calculated from voltage: i = vi / xf - norton equivalent!
        current = vi / self.par['xf'][0]
        
        return current
    
    def current_injections(self, x, v):
        i_n_r = self.par['S_n'][0] / self.sys_par['s_n'] 
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r
    
    def init_from_load_flow(self, x_0, v_0, S):
        """Initialize from load flow solution"""
        self._input_values['i_ref'] = self.par['i_ref'][0]
        
        # Initialize states
        X0 = self.local_view(x_0)
        X0['x_pi'][:] = 0.0  # Start with zero integral
        
        # Initialize the last current value
        self._last_current = 0.0

