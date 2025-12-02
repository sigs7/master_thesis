
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
            ['L1',      'B3',   5,    2.5,    'Z'],
        ],
        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['IB',      'B1',    10,    22,       0,      1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       1e-5,      1e-5,         1e5,      10000,          1e5,        1e5],
            ],
        },

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    10,   22,      1.0,     1.0,      0.5,    0.1,     0.1,    0.1,        0,          0.1   ] # enable perfect tracking: 1, else 0
            ],
        },

        'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'S_n', 'V_n',         'J_m',               'J_e',        'H_m',       'H_e',            'K',          'D',        'Kp_pitch',     'Ki_pitch',   'rho',   'R',  'P_rated', 'omega_m_rated', 'wind_rated'],
            ['WT1', 'UIC1',  10,    22,          155000000.,            160.,          1,           1,            2317025352.,    9240560.,    60*3.14/180,   13*3.14/180,   1.225,  89.15,     1.05,       9.6,               10.6]
                # [-,-, , , Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu (system base power), RPM, m/s] from OpenFAST
                # must have H_m and H_e placeholders for now
        ],
    }
    }

    """ 'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'H_m', 'H_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'rho', 'R', 'P_rated'],
            ['WT1', 'UIC1',  4.5,  0.9,  1.5,  1.5,    5.0,         0.5,      1.225,  89.15, 1.0]
                # [-,    -,   s,    s,   pu,   pu,   rad/pu,     rad/pu,    kg/m^3,    m,    pu]
        ],


        'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'H_m', 'H_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'rho', 'R', 'P_rated'],
            ['WT1', 'UIC1',  5,    1, 2317025352, 9240560,    60*3.14/180,   13*3.14/180,   1.225,  89.15, 1.0]
                # [-,-, , , Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu] from PF
        ],
    }
    } 
    'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'S_n', 'V_n',         'J_m',               'J_e',        'H_m',       'H_e',            'K',          'D',        'Kp_pitch',     'Ki_pitch',   'rho',   'R',  'P_rated', 'omega_m_rated', 'wind_rated'],
            ['WT1', 'UIC1',  15,    22,    3.50803121174E+08,      1.836784E+06,       1,           1,      6.97376449E+08, 7.1186519E+07,    60*3.14/180,   13*3.14/180,   1.225,  120.97,     1.05,       7.57,               10.6]
                # [-,-, , , Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu (system base power), RPM, m/s] from OpenFAST
                # must have H_m and H_e placeholders for now
        ],
    }
    
    """