from src.dyn_models.utils import DAEModel
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq
import os
import src.utility_functions as dps_uf

class WindTurbine(DAEModel):
    """
    'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'S_n', 'V_n',         'J_m',             'J_e',             'K',          'D',        'Kp_pitch',     'Ki_pitch',   'T_pitch', 'max_pitch', 'min_pitch', 'max_pitch_rate',     'rho',     'R',      'P_rated', 'omega_m_rated', 'wind_rated', 'N_gearbox','gb_gen_efficiency','MPT_filename', 'Cp_filename'],
            ['WT1', 'UIC1',  15,    22,          310619488.,        1836784,        697376449.,    71186519.,       0.66,           0.2,           0.1,         90,           0,           2,              1.225,    120.97,       1.0,       7.53,      10.6,           1,   0.95,           'MPT_Kopt2150.csv', 'Cp_Kopt2150.csv']
            # [-,     -,     MW,     kV,           kg m^2,           kg m^2,          Nm/rad,       Nms/rad,        rad/pu,         rad/pu,        s,            deg,         deg,         deg/s,          kg/m^3,     m,          pu,         RPM,        m/s, -,  eta_gb*eta_gen (0-1), -, -] 
                
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
        RPM_to_rad_per_s = 2 * np.pi / 60  # 1 RPM = 2π/60 rad/s
        self.par['omega_m_rated'] = self.par['omega_m_rated'] * RPM_to_rad_per_s
        
        self._debug_counter = 0
        
        # Load wind data from .hh file for variable wind speed
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
        #self._dt = 5e-3  # Timestep in seconds

        # convert all WT params to pu:
        w_m_base = self.par['omega_m_rated']  # rad/s
        T_base = self.par['S_n'] * 1e6 / w_m_base  # Nm
        
        # Calculate H_m and H_e from J_m and J_e as instance variables (arrays, one per unit)
        self.H_m = 0.5 * self.par['J_m'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.H_e = 0.5 * self.par['J_e'] * w_m_base**2 / (self.par['S_n'] * 1e6)
        self.par['K'] = self.par['K'] / T_base
        self.par['D'] = self.par['D'] * w_m_base / T_base
        self.par['max_pitch'] = self.par['max_pitch'] * np.pi / 180
        self.par['min_pitch'] = self.par['min_pitch'] * np.pi / 180
        self.par['max_pitch_rate'] = self.par['max_pitch_rate'] * np.pi / 180
        
    def connections(self):
        return [
            {
                'input': 'P_e',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'p_e',
            },
            {
                'input': 'S_n_UIC',
                'source': {
                    'container': 'vsc',
                    'mdl': '*',
                    'id': self.par['UIC'],
                },
                'output': 'S_n',
            },
            {
                'output': 'P_ref',
                'destination': {
                    'container': 'vsc',
                    'mdl': '*',
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
        P_aero = self.P_aero(x, v)
        Pe = self.P_e(x, v) * self.S_n_UIC(x, v) / par['S_n'] # UIC pu -> WT pu
        P_ref = self.P_ref(x, v) * self.S_n_UIC(x, v) / par['S_n']  # P_ref is in UIC local base 
        
        Tm = P_aero / X['omega_m'] if X['omega_m'] > 0 else 0
        # P_e is electrical output; mechanical power drawn from shaft = P_e/eta
        Pe_mech = Pe / par['gb_gen_efficiency'] if par['gb_gen_efficiency'] > 0 else Pe
        Te = Pe_mech / X['omega_e'] if X['omega_e'] > 0 else 0

        # shaft torque
        theta_s = X['theta_m'] - X['theta_e'] #/ par['N_gearbox']
        omega_s = X['omega_m'] - X['omega_e'] #/ par['N_gearbox']
        T_shaft = par['K'] * theta_s + par['D'] * omega_s
        
        # swing eqs for wt dynamics
        dX['omega_m'] = (1/self.H_m) * (Tm - T_shaft)
        dX['omega_e'] = (1/self.H_e) * (T_shaft - Te)# (T_shaft/par['N_gearbox'] - Te)
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']

        max_pitch = par['max_pitch'][0]
        min_pitch = par['min_pitch'][0]
        max_pitch_rate = par['max_pitch_rate'][0]
        omega_m_ref = 1.0 # 'hardcoded' as the rated speed from init willl always be 1 pu
        e_omega = X['omega_m'][0] - omega_m_ref
        pitch_reference = 0.0

        if e_omega < -0.05:
            # Region 2: MPPT (below rated) - reset integral
            dX_pitch_integral = 0.0
        else:  # Region 3: Power limiting (speed approaching rated)
            # Calculate controller output to check for anti-windup
            PIctrl_integral_term = par['Ki_pitch'][0] * X['pitch_PI_integral_state'][0]
            PIctrl_proportional_term = par['Kp_pitch'][0] * e_omega
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
        delta_pitch_angle = 1/par['T_pitch'][0] * (pitch_reference - X['pitch_angle'][0])
        delta_pitch_angle = np.clip(delta_pitch_angle, -max_pitch_rate, max_pitch_rate) # limit rate of change in pitch angle
        
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
            #print('dX[pitch_PI_integral_state]:', dX['pitch_PI_integral_state'])
            #print('  X[pitch_PI_integral_state]:', X['pitch_PI_integral_state'])
            #print('dX[pitch_angle]:', dX['pitch_angle'])
            print('  X[pitch_angle]:', X['pitch_angle'])
            #print('  pitch_angle:', self._pitch_angle)
            print('  P_aero (WT local pu):', P_aero)
            print('  Pe (WT local pu):', Pe)
            print('  P_ref (Wt local pu):', P_ref)
            print('cp: ', self.load_and_set_Cp(x, v))
            print('  Region:', 'MPPT')
            print('H_m:', self.H_m.astype(float))
            print('H_e:', self.H_e.astype(float))
            print('par[K]:', par['K'].astype(float))
            print('par[D]:', par['D'].astype(float))
            #print('  omega_m_ref:', omega_m_ref)

        return
    

    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        self._input_values['P_e'] = self.P_e(x_0, v_0)
        self._input_values['S_n_UIC'] = self.S_n_UIC(x_0, v_0)

        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        u_rated = float(np.asarray(par['wind_rated']).ravel()[0])
        u_start = float(np.asarray(self.wind_speed(x_0, v_0)).ravel()[0])
        #N = float(np.asarray(par['N_gearbox']).ravel()[0])
        K = float(np.asarray(par['K']).ravel()[0])

        # Solve P_aero(omega) = MPT(omega) for consistent init; fallback to u_start / u_rated
        if not hasattr(self, '_mpt_interp'):
            self._load_MPT_table()
        def _res(om):
            X['omega_m'] = om
            X['pitch_angle'] = 0.0
            return float(self.P_aero(x_0, v_0).ravel()[0]) - float(self._mpt_interp(om * w_rated))
        try:
            omega_m_init_pu = brentq(_res, 0.05, 1.0) # scipy.optimize.brentq to finds where _res function is 0 -> omega_m_init_pu
        except ValueError:
            omega_m_init_pu = np.clip(u_start / u_rated, 0.05, 1.0) # if brentq fails, use approximation start wind speed / rated wind speed
            print('Brentq omega_m init failed, using approximation start wind speed / rated wind speed')
        X['omega_m'] = omega_m_init_pu
        X['omega_e'] = omega_m_init_pu #* N
        X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))

        # Shaft twist init: theta_s = Tm/K so T_shaft = Tm at steady state, omega_e-omega_m = 0 at steady state
        P_aero_init = float(np.asarray(self.P_aero(x_0, v_0)).ravel()[0])
        Tm = P_aero_init / omega_m_init_pu if omega_m_init_pu > 0 else 0
        theta_s = Tm / K
        X['theta_m'] = theta_s
        X['theta_e'] = 0.0  # setting electrical angle as reference

        if omega_m_init_pu >= 0.99: 
            # Region 3: when initializing at rated speed, solve for pitch such that P_aero = P_rated
            min_pitch = float(np.asarray(par['min_pitch']).ravel()[0])
            max_pitch = float(np.asarray(par['max_pitch']).ravel()[0])
            Ki = float(np.asarray(par['Ki_pitch']).ravel()[0])
            X['omega_m'] = 1.0  # ensure rated for P_aero eval
            X['omega_e'] = 1.0 #N
            self.load_and_set_Cp(x_0, v_0)  # ensure Cp table loaded
            def _res_pitch(pitch_rad):
                X['pitch_angle'] = pitch_rad
                return float(self.P_aero(x_0, v_0).ravel()[0]) - 1.0
            try:
                pitch_eq = brentq(_res_pitch, min_pitch, max_pitch) # again brentq solves for where _res_pitch func is 0 -> pitch is the right val for P_aero = 1 pu
            except ValueError:
                P_at_min = _res_pitch(min_pitch) + 1.0 # fallback for when there is no solution for brentq (approx)
                pitch_eq = max_pitch if P_at_min > 1.0 else min_pitch # if P_aero at min pitch is higher than 1 pu, use max pitch, else use min pitch
                print('Brentq pitch init failed, using approximation min pitch or max pitch')
            pitch_eq = np.clip(pitch_eq, min_pitch, max_pitch)
            X['pitch_angle'] = pitch_eq
            X['pitch_PI_integral_state'] = pitch_eq / Ki if Ki > 0 else 0.0 # pitch_reference = Ki * integral + Kp * error, error = 0 in ss
            self._pitch_angle = pitch_eq
            # Region 3: P_aero = P_rated = 1.0, so recompute shaft twist
            P_aero = 1.0
            Tm = P_aero  # omega_m = 1 in ss region 3
            theta_s = Tm / K # shaft twist in ss region 3 (from Tm = K * theta_s + D * omega_s with omega_s = 0 in ss)
            X['theta_m'] = theta_s
        else:
            # Region 2: MPPT, pitch at minimum (typically 0)
            X['pitch_PI_integral_state'] = 0.0
            X['pitch_angle'] = max(0.0, float(np.asarray(par['min_pitch']).ravel()[0]))
            self._pitch_angle = float(np.asarray(X['pitch_angle']).ravel()[0])

        return

    def P_aero(self, x, v):
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.load_and_set_Cp(x, v)
        
        # Aerodynamic power from wind: P = 0.5 * rho * A * v^3 * Cp
        # A = pi * R^2 (swept area)
        P_aero_watts = 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp
        
        # Convert to per-unit using local base power
        S_base_watts = self.par['S_n'] * 1e6  # Convert MVA to Watts, WT local base
        P_aero_pu = P_aero_watts / S_base_watts
        
        return P_aero_pu  # WT pu
 
    def P_ref(self, x, v):
        X = self.local_view(x)
        par = self.par
        # Load MPT table on first call
        if not hasattr(self, '_mpt_interp'):
            self._load_MPT_table()
        
        # X['omega_m'] is in per-unit (base = omega_m_rated in rad/s)
        # Convert to rad/s for MPT table lookup (ensure scalars for interp)
        omega_m_pu = np.asarray(X['omega_m']).ravel()[0]
        omega_rated = np.asarray(self.par['omega_m_rated']).ravel()[0]
        rotor_speed_rad_s = float(omega_m_pu * omega_rated)
        P_mpt = float(self._mpt_interp(rotor_speed_rad_s))
        P_mpt_elec = par['gb_gen_efficiency'][0] * P_mpt  # electrical power reference (gearbox+gen losses)

        # Convert to system base for communication with VSC/UIC
        return P_mpt_elec * self.par['S_n'] / self.S_n_UIC(x, v) # WT pu -> UIC pu

    def P_ref_from_wind(self, wind_speed_mps, S_n_UIC):
        # used in init for UIC to avoid mismatch in power setpoints
        # finds the optimal power given a wind speed, assuming optimal tsr (using rated wind speed and rated rotor speed)
        if not hasattr(self, '_mpt_interp'):
            self._load_MPT_table()
        par = self.par
        R = float(np.asarray(par['R']).ravel()[0])
        w_rated = float(np.asarray(par['omega_m_rated']).ravel()[0])
        wind_rated = float(np.asarray(par['wind_rated']).ravel()[0])
        lam_ref = R * w_rated / wind_rated
        omega_rad = lam_ref * wind_speed_mps / R
        P_mpt = float(self._mpt_interp(omega_rad))
        P_mpt_elec = par['gb_gen_efficiency'][0] * P_mpt  # electrical power reference (gearbox+gen losses)
        S_n_WT = float(np.asarray(par['S_n']).ravel()[0])
        return P_mpt_elec * S_n_WT / float(np.asarray(S_n_UIC).ravel()[0])

    def _load_MPT_table(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        # Extract filename string from array (typically just one unit)
        mpt_filename = self.par['MPT_filename'][0] if isinstance(self.par['MPT_filename'], np.ndarray) else self.par['MPT_filename']
        path = os.path.join(project_root, 'wind_data', mpt_filename)
        
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

    def load_and_set_Cp(self, x, v):
        par = self.par
        X = self.local_view(x)
        
        # omega_m is stored in per-unit (base = omega_m_rated in rad/s)
        omega_m_rad_s = X['omega_m'] * par['omega_m_rated'] # pu speed * base speed -> rad/s
        wind_speed = self.wind_speed(x, v) # m/s
        # Handle array case: use np.where for element-wise conditional
        tip_speed_ratio = np.where(wind_speed > 0, omega_m_rad_s * par['R'] / wind_speed, 0)
        
        # Load Cp data if not already loaded
        if not hasattr(self, '_cp_data'):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            cp_filename = self.par['Cp_filename'][0] if isinstance(self.par['Cp_filename'], np.ndarray) else self.par['Cp_filename']
            path = os.path.join(project_root, 'wind_data', cp_filename)
            with open(path, 'r') as f:
                lines = f.readlines()
            pitch_line = lines[4].strip()
            if pitch_line.startswith('#'):
                pitch_line = pitch_line[1:].strip()
            pitch_angles = np.array([float(x) for x in pitch_line.split()])
            tsr_line = lines[6].strip()
            if tsr_line.startswith('#'):
                tsr_line = tsr_line[1:].strip()
            tip_speed_ratios = np.array([float(x) for x in tsr_line.split()])
            cp_start_idx = None
            for i, line in enumerate(lines):
                if '# Power coefficient' in line:
                    cp_start_idx = i + 1
                    break
            if cp_start_idx is None:
                raise ValueError("Could not find '# Power coefficient' section in Cp file")
            while cp_start_idx < len(lines) and not lines[cp_start_idx].strip():
                cp_start_idx += 1
            cp_values = []
            for i in range(len(tip_speed_ratios)):
                if cp_start_idx + i >= len(lines):
                    break
                line = lines[cp_start_idx + i].strip()
                if not line:
                    continue
                cp_row = np.array([float(x) for x in line.split() if x.strip()])
                cp_values.append(cp_row)
            if len(cp_values) > 0:
                expected_length = len(pitch_angles)
                cp_values_fixed = []
                for row in cp_values:
                    if len(row) > expected_length:
                        cp_values_fixed.append(row[:expected_length])
                    elif len(row) < expected_length:
                        padded = np.zeros(expected_length)
                        padded[:len(row)] = row
                        cp_values_fixed.append(padded)
                    else:
                        cp_values_fixed.append(row)
                cp_values = np.array(cp_values_fixed)
            else:
                raise ValueError("No Cp data found in Cp file")
            self._cp_interp = RegularGridInterpolator(
                (pitch_angles, tip_speed_ratios),
                cp_values.T,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._cp_data = True
        
        # Interpolate Cp value - pass as 1D array of length 2 for a single point
        # Convert to Python floats first to avoid any array issues
        tsr = float(tip_speed_ratio) if np.isscalar(tip_speed_ratio) else float(tip_speed_ratio.item())
        # Use state variable pitch_angle
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

    def wind_speed_init(self):
        """Wind speed at t=0 (m/s). No state needed. Use for power flow / init."""
        return 8.0

    def wind_speed(self, x, v):
        """Returns wind speed in m/s."""
        # Option: read from wind file - uncomment to use interpolated .hh data
        # t = self._debug_counter * self._dt
        # return float(self._wind_interp(t))
        u_wind = 8.0
        if hasattr(self, '_debug_counter') and self._debug_counter >= 24000:
            u_wind = 11.5
        return u_wind
