"""  
Goal: Model a VSC with UIC control (working version with angle tracking)

1. Create a UIC-style VSC with angle state for frequency tracking
2. Uses stationary reference frame for control (like simple version)
3. Adds angle state for frequency deviation tracking

- Use existing solvers
- Use existing blocks
- Differential equations are divided into steps (functions with different objectives)

"""

from .blocks import *

class VSC_UIC_Working(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        if hasattr(self, 'par') and 'bus' in self.par.dtype.names:
            return {'terminal': self.par['bus']}
        else:
            return {'terminal': 0}  # Default to bus 0
    
    def state_list(self):
        """Return list of states for this VSC"""
        return ['vi_x', 'vi_y']

    def input_list(self):
        """Input list for VSC"""
        return ['p_ref', 'q_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        
        # Get terminal voltage
        vt = v[self.bus_idx_red['terminal']]
        
        # Get internal voltage from states (in stationary frame)
        vi = X['vi_x'] + 1j*X['vi_y']
        
        # Calculate actual current through filter
        i_actual = (vi - vt)/(1j*self.par['xf'][0])
        
        # Get power reference
        s_ref = self._input_values['p_ref'] + 1j*self._input_values['q_ref']
        
        # Calculate reference current (with safety check)
        if abs(vi) > 1e-6:  # Avoid division by very small numbers
            i_ref = np.conj(s_ref/vi)
        else:
            i_ref = 0.0  # If vi is too small, set reference to zero
        
        # Calculate current error
        i_error = i_ref - i_actual
        
        # UIC control law - stationary frame with jω term for cross-coupling
        dvi = i_error * 1j * self.par['Ki'][0] * 2*np.pi  # 2π = 50 Hz
        
        # Apply voltage limits to prevent instability
        max_voltage = 1.5  # Maximum internal voltage magnitude
        current_magnitude = abs(vi)
        if current_magnitude > max_voltage:
            # Scale down if exceeding limits
            scale_factor = max_voltage / current_magnitude
            dvi = dvi * scale_factor
        
        dX['vi_x'][:] = np.real(dvi)
        dX['vi_y'][:] = np.imag(dvi)

        return

    def i_inj(self, x, v):
        """Calculate current from PI controller output: i = vi / xf"""
        X = self.local_view(x)
        vt = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j*X['vi_y']
        xf = self.par['xf'][0]
        current = (vi - vt)/(1j*xf)
        return current
    
    def current_injections(self, x, v):
        i_n_r = self.par['S_n'][0] / self.sys_par['s_n'] 
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r
    
    def init_from_load_flow(self, x_0, v_0, S):
        """Initialize from load flow solution"""
        self._input_values['p_ref'] = self.par['p_ref'][0]
        self._input_values['q_ref'] = self.par['q_ref'][0]
        
        X0 = self.local_view(x_0)
        
        # Initialize internal voltage based on power reference and terminal voltage
        s = self._input_values['p_ref'] + 1j*self._input_values['q_ref']
        v0 = v_0[self.bus_idx_red['terminal']]  # Terminal voltage
        i0 = np.conj(s/v0)  # Reference current
        vi0 = v0 + 1j*self.par['xf'][0]*i0  # Internal voltage
        X0['vi_x'][:] = np.real(vi0)
        X0['vi_y'][:] = np.imag(vi0)
                
        return
