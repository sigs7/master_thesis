
def load():
    return {
        'base_mva': 10,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
                ['name',    'V_n'],
                ['B1',      22],
                ['B2',      22]
        ],

        'lines': [
                ['name',    'from_bus',    'to_bus',    'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
                ['L1-2',    'B1',          'B2',        25,         10,    22,     'PF',       1e-4,   1e-3,   0.0],
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B2',   10,    5,    'Z'],
        ],

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B1',    10,   22,      1.0,     1.0,      0.5,    0.1,     0.1,    0.001,        1,          0.1   ] # enable perfect tracking: 1, else 0
            ],
        },
        'windturbine': {
        'WindTurbine': [
            ['name', 'UIC', 'V_n', 'H_m', 'H_e', 'K', 'D', 'Kp_pitch', 'Ki_pitch', 'rho', 'R', 'P_rated'],
            ['WT1', 'UIC1',  22,    5,    1, 2317025352, 9240560,    60*3.14/180,   13*3.14/180,   1.225,  89.15, 1.042]
                # [-,-, , , Nm/rad, Nms/rad, rad/pu, rad/pu, kg/m^3, m, pu] from PF
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
    } """