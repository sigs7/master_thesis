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
        if hasattr(self, 'par') and 'bus' in self.par.dtype.names:
            return {'terminal': self.par['bus']}
        else:
            return {'terminal': 0}  # Default to bus 0
    
    def state_list(self):
        """Return list of states for this VSC"""
        return ['vi_x', 'vi_y', 'angle']

    def input_list(self):
        """Input list for VSC"""
        return ['p_ref', 'q_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        
        # Get terminal voltage
        vt = v[self.bus_idx_red['terminal']]
        
        # Get internal voltage from states (in 50 Hz rotating reference frame)
        vi_rotating = X['vi_x'] + 1j*X['vi_y']
        angle = X['angle']
        
        # Transform to stationary frame for current calculation
        vi_stationary = vi_rotating * np.exp(1j*angle)
        
        # Calculate actual current in stationary frame
        i_actual_stationary = (vi_stationary - vt)/(1j*self.par['xf'][0])
        
        # Get power reference
        s_ref = self._input_values['p_ref'] + 1j*self._input_values['q_ref']
        
        # Calculate reference current in stationary frame (with safety check)
        if abs(vi_stationary) > 1e-6:  # Avoid division by very small numbers
            i_ref_stationary = np.conj(s_ref/vi_stationary)
        else:
            i_ref_stationary = 0.0  # If vi is too small, set reference to zero
        
        # Calculate current error in stationary frame
        i_error_stationary = i_ref_stationary - i_actual_stationary
        
        # UIC control law - work directly in stationary frame (like simple version)
        # The jω term creates cross-coupling for 50 Hz operation
        dvi_stationary = i_error_stationary * 1j * self.par['Ki'][0] * 2*np.pi
        
        # Transform back to rotating frame
        dvi_rotating = dvi_stationary * np.exp(-1j*angle)
        
        # Apply voltage limits to prevent instability
        max_voltage = 1.5  # Maximum internal voltage magnitude
        current_magnitude = abs(vi_rotating)
        if current_magnitude > max_voltage:
            # Scale down if exceeding limits
            scale_factor = max_voltage / current_magnitude
            dvi_rotating = dvi_rotating * scale_factor
        
        dX['vi_x'][:] = np.real(dvi_rotating)
        dX['vi_y'][:] = np.imag(dvi_rotating)
        
        # Update angle state - tracks frequency deviation from 50 Hz
        # For grid-forming operation, maintain constant 50 Hz
        dX['angle'][:] = 2*np.pi * self.sys_par['f_n']
        return

    def i_inj(self, x, v):
        """Calculate current injection in stationary frame"""
        X = self.local_view(x)
        vt = v[self.bus_idx_red['terminal']]
        
        # Convert 50 Hz rotating frame voltage to stationary frame
        vi_rotating = X['vi_x'] + 1j*X['vi_y']
        angle = X['angle']
        vi_stationary = vi_rotating * np.exp(1j*angle)
        
        xf = self.par['xf'][0]
        current = (vi_stationary - vt)/(1j*xf)
        
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
        
        # Initialize angle to grid angle (synchronized at start)
        X0['angle'][:] = np.angle(v0)
        
        return