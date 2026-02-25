def load():
    return {
        'base_mva': 1,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
            ['name',    'V_n'],
            ['B1',         1],
            ['B2',         1],
        ],

        'lines': [
            ['name',  'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit', 'R',    'X',   'B'],
            ['L1-2',        'B1',     'B2',          1,      1,      1,    'pu',   0,    0.1,     0],
        ],

        'generators': {
            'GEN': [
                ['name',   'bus',  'S_n',  'V_n',    'P',    'V',      'H',    'D',    'X_d',  'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['IB',      'B1',    1e6,    1,       0,      1,      1e5,      0,     1.05,   0.66,    0.328,      0.66,       1e-5,      1e-5,         1e5,      10000,          1e5,        1e5],
            ],
        },

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    1,      1,      1.0,     0.5,      0.0,    0.1/3,     0.0,    0.001,        1,          0.1   ] # enable perfect tracking: 1, else 0
            ],
        }, 

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B2',   1,    0.5,    'Z'],
        ]
    }

""" 'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    1,      1.0,     1.0,      0.5,    10,     0.0,    0.001,        True,          0.05   ]
            ],
        },  
        
    'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B2',    1,      1.0,     1.0,      0.5,    0.1,     0.1,    0.001,        False,          0.05   ]
            ],
        },
        

        """