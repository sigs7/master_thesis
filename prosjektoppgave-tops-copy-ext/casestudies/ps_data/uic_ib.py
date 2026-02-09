def load():
    return {
        'base_mva': 1,
        'f': 50,
        'slack_bus': 'B1',

        #'base_mva': system base/nominal complex power in [MVA].
        #'f': system frequency in [Hz].
        #'slack_bus': reference busbar with zero phase angle.

        'buses': [
            ['name',    'V_n'],
            ['B1',         10],
            ['B2',         10],
        ],

        #'V_n': base/nominal voltage in [kV].

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L1-2',        'B1',     'B2',          1,      1,      10,    'pu',   0,    0.1,     0],
        ],

        #'length': total line length in [km].
        #'S_n': base/nominal complex power in [MVA].
        #'V_n': base/nominal voltage in [kV].
        #'unit': chosen unit for jacobian admittance calculation (leave it as 'PF').
        #'R': line resistance in [Ohm/km].
        #'X': line reactance in [Ohm/km].
        #'B': line susceptance in [Ohm/km].


        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['IB',      'B1',    1e6,    10,       0,      1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       1e-5,      1e-5,         1e5,      10000,          1e5,        1e5],
            ],
        },

        'vsc': {
            'UIC': [
                ['name', 'bus', 'S_n', 'V_n', 'p_ref', 'q_ref',   'Ki',   'xf'],
                ['UIC1', 'B2',    1,      10,     0.0,      0.0,   0.01,    0.1],
            ],
        }
    }
