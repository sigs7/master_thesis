import os


def load():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Pick the FMU that actually exists (prefer OpenFAST/fast.fmu, otherwise fall back).
    fmu_candidates = [
        os.path.join(project_root, "OpenFAST", "fast.fmu"),
        os.path.join(project_root, "fast.fmu"),
    ]
    fmu_path = next((p for p in fmu_candidates if os.path.isfile(p)), fmu_candidates[0])

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
            ['L1',      'B3',   20,    5,    'Z'],
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
                ['UIC1', 'B2',    20,   22,      1.0,     0.5,      0.0,     0.05,     0.0,    0.1,        1,          0.1   ] # PQ bus for consistent q_ref=0 init
            ],
        },

        'FMUtoUICdrivetrain': {
            'FMUtoUICdrivetrain': [
                ['name', 'UIC', 'S_n', 'V_n', 'FMU_path', 'fmu_filename', 'control_mode', 'wd_path', 'openfast_test_dir',
                 'J_m', 'J_e', 'K', 'D', 'omega_m_rated', 'fmu_dt', 'ElecPwrCom_kW', 'efficiency'],
                ['FMUtoUICdrivetrain1', 'UIC1', 15, 22, fmu_path, 'fast.fmu', 3,
                 os.path.join(project_root, 'openfast_fmu', 'resources', 'wd.txt'),
                 # Directory that CONTAINS the OpenFAST case folders (e.g. test1002/).
                 # The FMU selects the case via the parameter testNr=1002.
                 project_root,
                 352460500., 1836784., 69737644900./100., 35697187.234657425, 7.559987120819503, 0.01, 20000.0, 0.95756],
            ],
        },
    }

