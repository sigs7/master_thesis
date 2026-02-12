
def load():
    return {
        'base_mva': 10,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
                ['name',    'V_n'],
                ['B1',      22],
                ['B2',      22],
                ['B3',      22]
        ],

        'lines': [
                ['name',    'from_bus',    'to_bus',    'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
                ['L1-2',    'B1',          'B2',        25,         10,    22,     'PF',       1e-4,   1e-3,   0.0],
                ['L2-3',    'B2',          'B3',        25,         10,    22,     'PF',       1e-4,   1e-3,   0.0]
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B3',   15,    5,    'Z'],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['IB',      'B1',    10e8,    22,       0,      1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       1e-5,      1e-5,         1e5,      10000,          1e5,        1e5],
            ],
        },

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    20,   22,      1.0,     0.5,      0.0,     0.1,     0.0,    0.1,        1,          0.1   ] # PQ bus for consistent q_ref=0 init
            ],
        },

        'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'S_n', 'V_n',         'J_m',             'J_e',             'K',          'D',        'Kp_pitch',     'Ki_pitch',   'T_pitch', 'max_pitch', 'min_pitch', 'max_pitch_rate',     'rho',     'R',      'P_rated', 'omega_m_rated', 'wind_rated', 'N_gearbox',  'gb_gen_efficiency','MPT_filename', 'Cp_filename'],
            ['WT1', 'UIC1',  15,    22,          310619488.,        1836784,        697376449.,    71186519.,       0.66,           0.2,           0.1,         90.0,           0.0,           2.0,              1.225,    120.97,       1.0,       7.53,      10.6,           1,   0.95,           'MPT_Kopt2150.csv', 'Cp_Ct_Cq.IEA15MW.ROSCO.txt']
            # [-,     -,     MW,     kV,           kg m^2,           kg m^2,          Nm/rad,       Nms/rad,        rad/pu,         rad/pu,        s,            deg,         deg,         deg/s,          kg/m^3,     m,          pu,         RPM,        m/s, -, -, -] 
                
        ],
    }
    }

""" 'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    20,   22,      1.0,     0.33,      0.0,    0.1,     0.0,    0.1,        1,          0.1   ] # enable perfect tracking: 1, else 0
            ],
        }, """