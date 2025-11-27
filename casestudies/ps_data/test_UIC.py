
def load():
    return {
        'base_mva': 1,
        'f': 50,
        'slack_bus': 'B1',

        'buses': [
                ['name',    'V_n'],
                ['B1',      1],
                ['B2',      1]
        ],

        'lines': [
                ['name',    'from_bus',    'to_bus',    'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
                ['L1-2',    'B1',          'B2',        25,         1,    1,     'PF',       1e-4,   1e-3,   0.0],
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B2',   1,    0.5,    'Z'],
        ],

        'vsc': {
            'UIC_sig': [
                ['name', 'bus', 'S_n', 'V_n', 'v_ref', 'p_ref', 'q_ref',   'Ki',   'Kv',    'xf', 'perfect_tracking', 'T_filter'],
                ['UIC1', 'B1',    1,      1,      1.0,     1.0,      0.5,    0.1,     0.0,    0.001,        0,          0.1   ] # enable perfect tracking: 1, else 0
            ],
        }
    }