from src.dyn_models.utils import DAEModel
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import os

class WindTurbine(DAEModel):

    """
    Inputs:
    'windturbine': {
        'WindTurbine': [
            [
                'name', 'UIC', 'H_m', 'H_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'rho', 'R', 'P_rated', omega_m_rated, wind_rated],
            [
                'WT1', 'UIC1',  1.0,    1.0, 2317025352, 9240560,    60*np.pi/180,   13*np.pi/180,   1.225,  89.15, 1.0, 1.0, 10]
                # [-,-, s, s, Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu, RPM, m/s] from PF
                # Note: omega_m_rated should be provided in RPM and will be converted to rad/s internally
        ],
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        sn = self.par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        self.par['S_n'] = sn
        self._sys_to_local = self.sys_par['s_n'] / self.par['S_n']
        self._local_to_sys = self.par['S_n'] / self.sys_par['s_n']
        # Convert omega_m_rated from RPM to rad/s
        # Input parameter should be provided in RPM, converted to rad/s for internal use
        # 1 RPM = 2π/60 rad/s = π/30 rad/s
        RPM_to_rad_per_s = 2 * np.pi / 60  
        self.par['omega_m_rated'] = self.par['omega_m_rated'] * RPM_to_rad_per_s
        
        # Initialize debug counter for wind speed changes
        # Counter increments each time state_derivatives is called
        self._debug_counter = 0
        
        # Load wind data from .hh file
        # File format: first line is number of columns, then time (col 1) and wind speed in m/s (col 2)
        wind_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'wind_data', '10mps_NTM_3xDTU10MW_IECKAI_VS_T1.hh')
        wind_data = np.loadtxt(wind_file_path, skiprows=1, usecols=(0, 1))
        wind_times = wind_data[:, 0]  # First column: time in seconds
        wind_speeds = wind_data[:, 1]  # Second column: wind speed in m/s
        
        # Create interpolation function for smooth wind speed transitions
        # Use linear interpolation for natural smooth transitions
        self._wind_interp = interp1d(wind_times, wind_speeds, kind='linear', 
                                     bounds_error=False, fill_value=(wind_speeds[0], wind_speeds[-1]))
        self._dt = 5e-3  # Timestep in seconds
        
        # w_e = p/2 * w_m -> electrical speed is much larger than mechanical if assuming poles are 84
        # convert all WT params to pu:
        # H_m = J_m * omega_synchronous^2 / (S_n * 1e6 * (p/2)**2)
        """ omega_norm = self.par['omega_m_rated']
        omega_synchronous = 2 * np.pi * self.sys_par['f_n']
        poles = 2 # unsure about this, based on rated speed (ns = 120*f/p)
        self.par['H_m'] = 0.5 * self.par['J_m'] * omega_norm**2 / (self.par['S_n'] * 1e6 * (poles/2)**2)
        # H_e = J_e * omega_synchronous^2 / (S_n * 1e6 * (p/2)**2)
        self.par['H_e'] = 0.5 * self.par['J_e'] * omega_synchronous**2 / (self.par['S_n'] * 1e6 * (poles/2)**2)
        # K = K * omega_synchronous / (S_n * 1e6 * (p/2)**2)
        self.par['K'] = self.par['K'] * omega_norm / (self.par['S_n'] * 1e6 * (poles/2)**2)
        # D = D * omega_synchronous / (S_n * 1e6 * (p/2)**2)
        self.par['D'] = self.par['D'] * omega_norm**2 / (self.par['S_n'] * 1e6 * (poles/2)**2) """

        w_base = self.par['omega_m_rated'] # in rad/s
        T_base = self.par['S_n'] * 1e6 / w_base # W / rad/s = Nm
        
        self.par['H_m'] = 0.5 * self.par['J_m'] * w_base**2 / (self.par['S_n'] * 1e6)
        self.par['H_e'] = 0.5 * self.par['J_e'] * w_base**2 / (self.par['S_n'] * 1e6)
        self.par['K'] = self.par['K'] / T_base
        self.par['D'] = self.par['D'] * w_base / T_base
        
    def connections(self):
        return [
            {
                'input': 'P_e',
                'source': {
                    'container': 'vsc',
                    'mdl': 'UIC_sig',
                    'id': self.par['UIC'],
                },
                'output': 'p_e',
            },
            {
                'input': 'S_n_UIC',
                'source': {
                    'container': 'vsc',
                    'mdl': 'UIC_sig',
                    'id': self.par['UIC'],
                },
                'output': 'S_n',
            },
            {
                'output': 'P_ref',
                'destination': {
                    'container': 'vsc',
                    'mdl': 'UIC_sig',
                    'id': self.par['UIC'],
                },
                'input': 'p_ref',
            }
        ]

    def state_list(self):
        return ['omega_m', 'omega_e', 'theta_m', 'theta_e', 'pitch_PI_integral_state', 'pitch_angle']

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par
        Pm = self.P_m(x, v)
        Pe = self.P_e(x, v) * self.S_n_UIC(x, v) / par['S_n'] # UIC pu -> WT pu
        P_rated = par['P_rated']
        P_ref = self.P_ref(x, v) * self.S_n_UIC(x, v) / par['S_n']  # P_ref is in UIC local base 
        
        Tm = Pm / X['omega_m'] if X['omega_m'] > 0 else 0
        Te = Pe / X['omega_e'] if X['omega_e'] > 0 else 0
        omega_e_mech_base = X['omega_e']
        
        # swing eqs for wt dynamics
        dX['omega_m'] = (1/par['H_m']) * (Tm - (par['K'] * (X['theta_m'] - X['theta_e']) + par['D'] * (X['omega_m'] - omega_e_mech_base)))
        dX['omega_e'] = (1/par['H_e']) * ((par['K'] * (X['theta_m'] - X['theta_e']) + par['D'] * (X['omega_m'] - omega_e_mech_base)) - Te)
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']

        # TODO Inputs for current WT?
        max_pitch = 1.57  # 90 degrees max
        min_pitch = 0.0  # 0 degrees min 
        max_pitch_rate = 0.03490  # 2 deg/s max rate
        omega_m_ref = 1.0  # Rated speed in pu
        e_omega = X['omega_m'] - omega_m_ref
        pitch_reference = 0.0

        if e_omega < -0.05:
            # Region 2: MPPT (below rated) - reset integral
            dX_pitch_integral = 0.0
        else:  # Region 3: Power limiting (speed approaching rated)
            # Calculate controller output to check for anti-windup
            PIctrl_integral_term = par['Ki_pitch'] * X['pitch_PI_integral_state']
            PIctrl_proportional_term = par['Kp_pitch'] * e_omega
            pitch_reference_unclamped = PIctrl_integral_term + PIctrl_proportional_term
            
            # Anti-windup -> stops integration term when reference is at limit to prevent over- and undershoots
            if pitch_reference_unclamped >= max_pitch or pitch_reference_unclamped <= min_pitch:
                dX_pitch_integral = 0.0  # Stop integrating when output hits limits
            else:
                dX_pitch_integral = e_omega  # Normal integration
            
            # Clamp pitch_reference to max and min pitch angle
            pitch_reference = np.clip(pitch_reference_unclamped, min_pitch, max_pitch)
        
        # update integral state of PI control for pitch reference
        dX['pitch_PI_integral_state'] = dX_pitch_integral
        
        ## Servo  
        delta_pitch_angle = 1/par['T_pitch'] * (pitch_reference - X['pitch_angle'])
        delta_pitch_angle = np.clip(delta_pitch_angle, -max_pitch_rate, max_pitch_rate) # limit rate of change in pitch angle
        
        """ # this should not be necessary as the checks above should prevent this
        if X['pitch_angle'] >= max_pitch and delta_pitch_angle > 0:
            delta_pitch_angle = 0.0  # Stop if at upper limit and trying to increase
        elif X['pitch_angle'] <= min_pitch and delta_pitch_angle < 0:
            delta_pitch_angle = 0.0  # Stop if at lower limit and trying to decrease """
        
        dX['pitch_angle'] = delta_pitch_angle
        
        self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 2 or self._debug_counter == 3 or self._debug_counter == 4 or self._debug_counter == 5 or self._debug_counter == 6 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            print('Debug values (iteration', self._debug_counter, '):')
            print('  X[omega_m]:', X['omega_m'])
            print('  X[omega_e]:', X['omega_e'])
            print('  X[theta_m]:', X['theta_m'])
            print('  X[theta_e]:', X['theta_e'])
            print('dX[omega_m]:', dX['omega_m'])
            print('dX[omega_e]:', dX['omega_e'])
            print('dX[theta_m]:', dX['theta_m'])
            print('dX[theta_e]:', dX['theta_e'])
            print('dX[pitch_PI_integral_state]:', dX['pitch_PI_integral_state'])
            print('  X[pitch_PI_integral_state]:', X['pitch_PI_integral_state'])
            print('dX[pitch_angle]:', dX['pitch_angle'])
            print('  X[pitch_angle]:', X['pitch_angle'])
            print('  pitch_angle:', self._pitch_angle)
            print('  Pm (WT local pu):', Pm)
            print('  Pe (WT local pu):', Pe)
            print('  P_ref (Wt local pu):', P_ref)
            print('cp: ', self.Cp(x, v))
            print('  Region:', 'MPPT' if P_ref < (P_rated - 0.02) else 'Power Limiting')
            print('par[H_m]:', par['H_m'].astype(float))
            print('par[H_e]:', par['H_e'].astype(float))
            print('par[K]:', par['K'].astype(float))
            print('par[D]:', par['D'].astype(float))
            #print('  omega_m_ref:', omega_m_ref)

        return
    
    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        # P_e comes from UIC in UIC LOCAL BASE (not system base)
        # S_n_UIC comes from UIC to know its base for conversion
        self._input_values['P_e'] = self.P_e(x_0, v_0)  # UIC local base
        self._input_values['S_n_UIC'] = self.S_n_UIC(x_0, v_0)  # UIC's S_n for base conversion
        poles = 2 # change this later TODO
        ideal_tsr_start = par['R'] * 1 / par['wind_rated']
        start_omega_m_init = ideal_tsr_start * self.wind_speed(x_0, v_0) / (par['R']) # in rad/s
        
        # Initialize at a reasonable operating speed (e.g., 0.8 pu of rated)
        # This ensures we're in a valid operating region for MPT table lookup
        omega_m_init_pu = start_omega_m_init / par['omega_m_rated']  # in pu
        X['omega_m'] = omega_m_init_pu  # per-unit on mechanical base
        # Electrical speed: omega_e = omega_m * (poles/2) in same per-unit base
        # Both are normalized by omega_m_rated, so omega_e_pu = omega_m_pu * (poles/2)
        X['omega_e'] = omega_m_init_pu * (poles/2)  # per-unit (normalized by omega_m_rated)
        X['theta_m'] = 0.0
        X['theta_e'] = 0.0
        X['pitch_PI_integral_state'] = 0.0
        self._pitch_angle = 0.0

        return

    def P_m(self, x, v):
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.Cp(x, v)
        
        # Mechanical power from wind: P = 0.5 * rho * A * v^3 * Cp
        # A = pi * R^2 (swept area)
        P_m_watts = 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp
        
        # Convert to per-unit using local base power
        S_base_watts = self.par['S_n'] * 1e6  # Convert MVA to Watts, WT local base
        P_m_pu = P_m_watts / S_base_watts
        
        return P_m_pu # WT pu
 
    def P_ref(self, x, v):
        X = self.local_view(x)
        # Load MPT table on first call
        if not hasattr(self, '_mpt_interp'):
            self._load_MPT_table()
        
        # Initialize output array for all units (local base)
        P_ref_array = np.zeros(self.n_units)
        P_rated = self.par['P_rated']
        
        # Calculate for each unit (typically just 1 wind turbine)
        for i in range(self.n_units):
            # X['omega_m'] is in per-unit (base = omega_m_rated in rad/s)
            # Convert to rad/s for MPT table lookup
            rotor_speed_rad_s = X['omega_m'][i] * self.par['omega_m_rated'][i] # pu speed * base speed -> rad/s
            
            # Lookup optimal power from MPPT curve (expects rotor speed in rad/s)
            P_mpt = float(self._mpt_interp(rotor_speed_rad_s))
            #P_rated_low_wind = self.P_below_rated_wind(x, v)

            # Apply rated power limit (acts as natural transition to Region 3)
            P_ref_array[i] = min(P_mpt, P_rated[i]) # limit elec power demand
        
        # Convert to system base for communication with VSC/UIC
        return P_ref_array * self.par['S_n'] / self.S_n_UIC(x, v) # WT pu -> UIC pu


    def _load_MPT_table(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        path = os.path.join(project_root, 'wind_data', 'MPT_Kopt2150.csv')
        
        data = np.loadtxt(path, delimiter='\t')
        
        # Column 0: rotor speeds in RPM (will be converted to rad/s)
        # Column 1: optimal power (pu)
        rotor_speed_RPM = data[2:, 0]  # Skip header rows, in RPM
        optimal_powers = data[2:, 1]
        
        # Convert rotor speeds from RPM to rad/s
        # 1 RPM = 2π/60 rad/s = π/30 rad/s
        RPM_to_rad_per_s = 2 * np.pi / 60
        rotor_speed_rad_s = rotor_speed_RPM * RPM_to_rad_per_s
        
        # Create 1D interpolator (expects rotor speed in rad/s)
        from scipy.interpolate import interp1d
        self._mpt_interp = interp1d(
            rotor_speed_rad_s, 
            optimal_powers,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, optimal_powers[-1])  # Extrapolate: 0 below, max above
        )

    def Cp(self, x, v):
        par = self.par
        X = self.local_view(x)
        
        # Calculate tip speed ratio
        # omega_m is stored in per-unit (base = omega_m_rated in rad/s)
        # Convert to rad/s for TSR calculation
        omega_m_rad_s = X['omega_m'] * par['omega_m_rated'] # pu speed * base speed -> rad/s
        wind_speed = self.wind_speed(x, v) # m/s
        # Handle array case: use np.where for element-wise conditional
        tip_speed_ratio = np.where(wind_speed > 0, omega_m_rad_s * par['R'] / wind_speed, 0)
        
        # Load ROSCO format Cp data if not already loaded
        if not hasattr(self, '_cp_data'):
            # Get the path relative to the project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            path = os.path.join(project_root, 'wind_data', 'Cp_Ct_Cq.IEA15MW.ROSCO.txt')
            
            # Read ROSCO format file
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # Parse pitch angles from line 5 (index 4, 0-based)
            # Format: space-separated values, comment starts with #
            pitch_line = lines[4].strip()
            if pitch_line.startswith('#'):
                pitch_line = pitch_line[1:].strip()
            pitch_angles = np.array([float(x) for x in pitch_line.split()])  # In degrees
            
            # Parse TSR values from line 7 (index 6, 0-based)
            tsr_line = lines[6].strip()
            if tsr_line.startswith('#'):
                tsr_line = tsr_line[1:].strip()
            tip_speed_ratios = np.array([float(x) for x in tsr_line.split()])
            
            # Find the start of Cp data section (line with "# Power coefficient" comment)
            cp_start_idx = None
            for i, line in enumerate(lines):
                if '# Power coefficient' in line:
                    cp_start_idx = i + 1
                    break
            
            if cp_start_idx is None:
                raise ValueError("Could not find '# Power coefficient' section in ROSCO file")
            
            # Skip blank lines after the comment
            while cp_start_idx < len(lines) and not lines[cp_start_idx].strip():
                cp_start_idx += 1
            
            # Read Cp matrix (26 rows × 36 columns)
            # Each row corresponds to a TSR value, each column to a pitch angle
            cp_values = []
            for i in range(len(tip_speed_ratios)):
                if cp_start_idx + i >= len(lines):
                    break
                line = lines[cp_start_idx + i].strip()
                if not line:  # Skip empty lines
                    continue
                # Split and filter out empty strings from trailing whitespace
                cp_row = np.array([float(x) for x in line.split() if x.strip()])
                cp_values.append(cp_row)
            
            # Ensure all rows have the same length (should be 36 for pitch angles)
            if len(cp_values) > 0:
                expected_length = len(pitch_angles)
                # Trim or pad rows to expected length
                cp_values_fixed = []
                for row in cp_values:
                    if len(row) > expected_length:
                        cp_values_fixed.append(row[:expected_length])
                    elif len(row) < expected_length:
                        # Pad with zeros if shorter (shouldn't happen, but handle it)
                        padded = np.zeros(expected_length)
                        padded[:len(row)] = row
                        cp_values_fixed.append(padded)
                    else:
                        cp_values_fixed.append(row)
                cp_values = np.array(cp_values_fixed)  # Shape: (n_tsr, n_pitch)
            else:
                raise ValueError("No Cp data found in ROSCO file")
            
            # Create interpolation function using RegularGridInterpolator
            # ROSCO file: pitch angles are x-axis (columns), TSR are y-axis (rows)
            # cp_values[i, j] where i=TSR row, j=pitch column, so cp_values[i, j] = (TSR[i], pitch[j])
            # RegularGridInterpolator expects: (x_coords, y_coords) where x=pitch (columns), y=TSR (rows)
            # So we need to transpose cp_values so that values[i, j] = (pitch[i], TSR[j])
            # Or pass grid as (pitch_angles, tip_speed_ratios) and transpose cp_values
            self._cp_interp = RegularGridInterpolator(
                (pitch_angles, tip_speed_ratios),  # x=pitch (columns), y=TSR (rows)
                cp_values.T,  # Transpose so values[i, j] = (pitch[i], TSR[j])
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            self._cp_data = True
        
        # Interpolate Cp value - pass as 1D array of length 2 for a single point
        # Convert to Python floats first to avoid any array issues
        tsr = float(tip_speed_ratio) if np.isscalar(tip_speed_ratio) else float(tip_speed_ratio.item())
        # Use state variable pitch_angle (not _pitch_angle instance variable)
        X = self.local_view(x)
        pitch_angle_val = X['pitch_angle'] 
        pa = float(pitch_angle_val*180/np.pi) if np.isscalar(pitch_angle_val) else float(pitch_angle_val.item()*180/np.pi)
        
        # Clamp values to be within grid bounds to avoid extrapolation returning fill_value (0)
        # Grid order is (pitch_angles, tip_speed_ratios), so grid[0] = pitch, grid[1] = TSR
        pa_clamped = np.clip(pa, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
        tsr_clamped = np.clip(tsr, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())
        
        # Point order must match grid order: (pitch, TSR)
        point = np.array([pa_clamped, tsr_clamped], dtype=np.float64)
        # RegularGridInterpolator returns an array, so we need to extract the scalar
        Cp_table = float(self._cp_interp(point)[0])
        
        # Cp is dimensionless - return the coefficient directly
        return Cp_table

    def wind_speed(self, x, v):
        """Returns wind speed in m/s from interpolated wind data file.
        
        Uses the first two columns of the .hh wind data file:
        - Column 1: time in seconds
        - Column 2: wind speed in m/s
        
        Interpolates smoothly between data points using timestep 5e-3 seconds.
        """
        """ # Calculate current simulation time from counter and timestep
        t = self._debug_counter * self._dt
        
        # Interpolate wind speed at current time for smooth transitions
        u_wind = float(self._wind_interp(t))
 """
        u_wind = 8.0

        # Change wind speed after a certain number of iterations
        # Counter is initialized in __init__ and incremented in state_derivatives
        if hasattr(self, '_debug_counter') and self._debug_counter >= 24000:
            u_wind = 11.5
        
        return u_wind      
    
