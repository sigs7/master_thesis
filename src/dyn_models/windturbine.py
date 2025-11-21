from src.dyn_models.utils import DAEModel
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

class WindTurbine(DAEModel):

    """
    Inputs:
    'windturbine': {
        'WindTurbine': [
            [
                'name', 'UIC', 'H_m', 'H_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'rho', 'R', 'P_rated'],
            [
                'WT1', 'UIC1',  1.0,    1.0, 2317025352, 9240560,    60*np.pi/180,   13*np.pi/180,   1.225,  89.15, 1.0]
                # [-,-, s, s, Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu] from PF
        ],
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Convert K and D from physical units to per-unit
        # K: [Nm/rad] -> [pu], D: [Nm·s/rad] -> [pu]
        omega_base = 1.0  # rad/s (base angular velocity)
        S_base = self.sys_par['s_n'] * 1e6  # Convert MVA to Watts
        T_base = S_base / omega_base  # Torque base [Nm]
        
        # Convert stiffness and damping to per-unit
        self.par['K'] = self.par['K'] / T_base  # K_pu = K / T_base
        self.par['D'] = self.par['D'] * omega_base / T_base  # D_pu = D * omega_base / T_base

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
        return ['omega_m', 'omega_e', 'theta_m', 'theta_e', 'pitch_angle']

    def input_list(self):
        return ['P_e']

    def output_list(self):
        return ['P_ref']
    
    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        Pm = self.P_m(x, v)
        Pe = self.P_e(x, v)
        P_rated = par['P_rated']
        P_ref = self.P_ref(x, v)
        Tm = Pm/X['omega_m'] if abs(X['omega_m']) > 0.01 else 0
        Te = Pe/X['omega_e'] if abs(X['omega_e']) > 0.01 else 0
        
        # swing eqs for wt dynamics
        dX['omega_m'] = (1/par['H_m']) * (Tm - par['K'] * (X['theta_m'] - X['theta_e']) - par['D'] * (X['omega_m'] - X['omega_e']))
        dX['omega_e'] = (1/par['H_e']) * (Te + par['K'] * (X['theta_m'] - X['theta_e']) + par['D'] * (X['omega_m'] - X['omega_e']))
        dX['theta_m'] = X['omega_m']
        dX['theta_e'] = X['omega_e']


        if Pm < P_rated:  # Region 2: MPPT
            # Keep pitch at optimal, 0?
            dX['pitch_angle'] = par['Ki_pitch'] * (0.0 - X['pitch_angle'])
        else:  # Region 3: Power limiting
            # Active pitch control to regulate speed
            omega_m_ref = Pe/(Pm/X['omega_m']) if Pm > 0.01 else X['omega_m']
            dX['pitch_angle'] = par['Ki_pitch'] * (omega_m_ref - X['omega_m'])


        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter == 1 or self._debug_counter == 7500 or self._debug_counter == 100 or self._debug_counter == 500 or self._debug_counter == 1000 or (self._debug_counter % 5000 == 0 and self._debug_counter <= 60000): 
            print('Debug values (iteration', self._debug_counter, '):')
            print('  X[omega_m]:', X['omega_m'])
            print('  X[omega_e]:', X['omega_e'])
            print('  X[theta_m]:', X['theta_m'])
            print('  X[theta_e]:', X['theta_e'])
            print('dX[omega_m]:', dX['omega_m'])
            print('dX[omega_e]:', dX['omega_e'])
            print('dX[theta_m]:', dX['theta_m'])
            print('dX[theta_e]:', dX['theta_e'])
            print('  X[pitch_angle]:', X['pitch_angle'])
            print('  Pm:', Pm)
            print('  Pe:', Pe)
            print('  P_ref:', P_ref)
            print('  P_rated:', P_rated)
            print('cp: ', self.Cp(x, v))
            print('  Region:', 'MPPT' if P_ref[0] < (P_rated - 0.02) else 'Power Limiting')

        return
    
    def init_from_connections(self, x_0, v_0, S):
        X = self.local_view(x_0)
        par = self.par
        self._input_values['P_e'] = self.P_e(x_0, v_0)
        tip_speed_ratio = 7.0 # TODO
        
        X['theta_m'] = 0.0
        X['theta_e'] = 0.0
        
        # Initialize omega in rad/s, then convert to per-unit
        omega_base = 1.0  # rad/s - should have rated speed ??
        omega_m_rad_s = tip_speed_ratio * self.wind_speed(x_0, v_0) / par['R']
        X['omega_m'] = omega_m_rad_s / omega_base  # Convert to per-unit
        X['omega_e'] = X['omega_m'] # rated soeed here too?
        X['pitch_angle'] = 0.0 ## maybe start at other angle

        return

    def P_m(self, x, v):
        par = self.par
        wind_speed = self.wind_speed(x, v)
        Cp = self.Cp(x, v)
        
        # Mechanical power from wind: P = 0.5 * rho * A * v^3 * Cp
        # A = pi * R^2 (swept area)
        P_m_watts = 0.5 * par['rho'] * np.pi * par['R']**2 * wind_speed**3 * Cp
        
        # Convert to per-unit using system base power
        # sys_par['s_n'] is in MVA (from model['base_mva'])
        S_base_watts = self.sys_par['s_n'] * 1e6  # Convert MVA to Watts
        P_m_pu = P_m_watts / S_base_watts
        
        return P_m_pu


    def P_ref(self, x, v):
        # Load MPT table on first call
        if not hasattr(self, '_mpt_interp'):
            self._load_MPT_table()
        
        # Initialize output array for all units
        P_ref_array = np.zeros(self.n_units)
        P_rated = self.par['P_rated']
        
        # Calculate for each unit (typically just 1 wind turbine)
        for i in range(self.n_units):
            wind_speed = self.wind_speed(x, v)  # This returns scalar for now
            
            # Lookup optimal power from MPPT curve
            P_mpt = float(self._mpt_interp(wind_speed))
            
            # Apply rated power limit (acts as natural transition to Region 3)
            P_ref_array[i] = min(P_mpt, P_rated) # limit elec power demand
        
        return P_ref_array

    """ def P_ref(self, x, v):

        return self.P_m(x, v) """

    def _load_MPT_table(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        path = os.path.join(project_root, 'wind_data', 'MPT.csv')
        
        data = np.loadtxt(path, delimiter='\t')
        
        # Column 0: wind speeds (m/s)
        # Column 1: optimal power (pu)
        wind_speeds = data[2:, 0]  # Skip header rows
        optimal_powers = data[2:, 1]
        
        # Create 1D interpolator
        from scipy.interpolate import interp1d
        self._mpt_interp = interp1d(
            wind_speeds, 
            optimal_powers,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, optimal_powers[-1])  # Extrapolate: 0 below, max above
        )

    def Cp(self, x, v):
        par = self.par
        X = self.local_view(x)
        
        # Calculate tip speed ratio
        # omega_m is in per-unit (base = 1.0 rad/s), convert to rad/s for TSR calculation
        omega_base = 1.0  # rad/s
        omega_m_rad_s = X['omega_m'] * omega_base
        wind_speed = self.wind_speed(x, v)
        tip_speed_ratio = omega_m_rad_s * par['R'] / wind_speed if wind_speed > 0 else 0
        
        # Load CSV data if not already loaded
        if not hasattr(self, '_cp_data'):
            # Get the path relative to the project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            path = os.path.join(project_root, 'wind_data', 'cp.csv')
            
            # Read CSV file (tab-delimited)
            data = np.loadtxt(path, delimiter='\t')
            
            # Row 1 (index 0) is header/metadata - skip it
            # Row 2 (index 1) contains tip speed ratios (all columns, but first column is not used)
            # Rows 3+ (index 2+) contain: column 0 = pitch angle, columns 1+ = Cp values
            # Skip first column of tip speed ratios to align with cp_values columns
            tip_speed_ratios = data[1, 1:]  # Skip first column to match cp_values
            cp_pitch_angles = data[2:, 0]  # First column contains pitch angles
            cp_values = data[2:, 1:]  # Remaining columns contain Cp values
            
            # Create interpolation function using RegularGridInterpolator
            # RegularGridInterpolator expects: (grid_points_tuple, values, method='linear', bounds_error=False, fill_value=0.0)
            # grid_points_tuple: (x_coords, y_coords) where x = tip_speed_ratios, y = cp_pitch_angles
            # values: 2D array where values[i, j] corresponds to (cp_pitch_angles[i], tip_speed_ratios[j])
            # But our cp_values[i, j] corresponds to (cp_pitch_angles[i], tip_speed_ratios[j]) which is correct
            self._cp_interp = RegularGridInterpolator(
                (tip_speed_ratios, cp_pitch_angles), 
                cp_values.T,  # Transpose because RegularGridInterpolator expects (x, y) order
                method='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            self._cp_data = True
        
        # Interpolate Cp value - pass as 1D array of length 2 for a single point
        # Convert to Python floats first to avoid any array issues
        tsr = float(tip_speed_ratio) if np.isscalar(tip_speed_ratio) else float(tip_speed_ratio.item())
        pa = float(X['pitch_angle']) if np.isscalar(X['pitch_angle']) else float(X['pitch_angle'].item())
        
        # Clamp values to be within grid bounds to avoid extrapolation returning fill_value (0)
        tsr_clamped = np.clip(tsr, self._cp_interp.grid[0].min(), self._cp_interp.grid[0].max())
        pa_clamped = np.clip(pa, self._cp_interp.grid[1].min(), self._cp_interp.grid[1].max())
        
        point = np.array([tsr_clamped, pa_clamped], dtype=np.float64)
        # RegularGridInterpolator returns an array, so we need to extract the scalar
        Cp_table = float(self._cp_interp(point)[0])
        
        # Cp is dimensionless - return the coefficient directly
        return Cp_table

    """ def Cp(self, x, v):
        X = self.local_view(x)
        par = self.par
        c1 = par['c1']
        c2 = par['c2']
        c3 = par['c3']
        c4 = par['c4']
        c5 = par['c5']
        c6 = par['c6']
        x_param = par['x_param']

        lam = X['omega_m']*par['R']/self.wind_speed(x, v)
        big_lam = 1/(lam+0.08*X['pitch_angle']) - 0.035/(1+X['pitch_angle']**3)

        Cp = c1 * (c2 * 1/(big_lam) - c3 * X['pitch_angle'] - c4 * X['pitch_angle']**x_param - c5) * np.exp(- c6 * (1/big_lam)) 
        return Cp """

    def wind_speed(self, x, v):
        ## change to read from file
        u_wind = 14
        return u_wind

