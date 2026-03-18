import os
import numpy as np
from src.dyn_models.utils import DAEModel
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


class FMUtoUIC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ 
        'FMUtoUIC': {
            'FMUtoUIC': [
                ['name', 'FMU_path', 'UIC', 'fmu_filename'],
                ['FMUtoUIC1', 'FMU_path1', 'UIC1', 'fmu_filename1'],
            ],
        }
        """
        par = self.par
        fmu_filename = str(np.atleast_1d(par['fmu_filename']).ravel()[0])
        model_description = read_model_description(fmu_filename, validate=False)

        vrs = {}
        for variable in model_description.modelVariables:
            vrs[variable.name] = variable.valueReference

        print("Value References: \n")
        for name, vr in vrs.items():
            print(f"Variable: {name}, Value Reference: {vr}")

        unzipdir = extract(fmu_filename)

        wd_file_path = str(np.atleast_1d(par['wd_path']).ravel()[0])
        new_directory = str(np.atleast_1d(par['openfast_test_dir']).ravel()[0])

        os.makedirs(os.path.dirname(wd_file_path), exist_ok=True)
        with open(wd_file_path, 'w') as f:
            f.write(new_directory)

        fmu = FMU2Slave(guid=model_description.guid,
                        unzipDirectory=unzipdir,
                        modelIdentifier=model_description.coSimulation.modelIdentifier,
                        instanceName='instance1')
        fmu.instantiate()
        fmu.setReal([vrs['testNr']], [1002])  # int like DynaWind
        control_mode = int(np.atleast_1d(par['control_mode']).ravel()[0]) if 'control_mode' in par.dtype.names else 3
        fmu.setReal([vrs['Mode']], [control_mode])
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

        self.fmu = fmu
        self.vrs = vrs
        self._fmu_time = 0.0
        self._fmu_dt = float(np.atleast_1d(par['fmu_dt']).ravel()[0]) if 'fmu_dt' in par.dtype.names else 0.01

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
        return []

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def init_from_connections(self, x_0, v_0, S):
        self._input_values["P_e"] = self.P_e(x_0, v_0)
        self._input_values["S_n_UIC"] = self.S_n_UIC(x_0, v_0)

    def state_derivatives(self, dx, x, v):
        return

    # FMU output names from modelDescription.xml (causality="output")
    FMU_OUTPUT_NAMES = [
        'Time', 'HSShftTq', 'GenTq', 'Wind1VelX', 'RtVAvgxh', 'BldPitch1',
        'NacYaw', 'RefGenSpd', 'GenSpeed', 'RotSpeed', 'LSSGagPxa', 'Azimuth',
        'GenAccel', 'YawBrTAxp', 'YawBrTAyp',
    ]

    def get_fmu_outputs(self):
        # return (GenTq, GenSpeed) for current FMU time. Use for logging/plots.
        return self.fmu.getReal([self.vrs['GenTq'], self.vrs['GenSpeed']])

    def get_all_fmu_outputs(self):
        """Return dict of all FMU outputs (from modelDescription.xml)."""
        names = [n for n in self.FMU_OUTPUT_NAMES if n in self.vrs]
        if not names:
            return {}
        vrefs = [self.vrs[n] for n in names]
        vals = self.fmu.getReal(vrefs)
        return dict(zip(names, vals))

    def P_ref(self, x, v):
        # Read OpenFAST outputs (read-only)
        GenTq_kNm, GenSpeed_rpm = self.fmu.getReal([self.vrs['GenTq'], self.vrs['GenSpeed']])
        # Convert units
        omega = GenSpeed_rpm * 2.0 * np.pi / 60.0              # rad/s
        P_kW  = (GenTq_kNm * omega)                            # since kN·m * rad/s / 1 = kW
        # Convert to pu on S_n (if your UIC expects pu)
        S_n_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        p_ref_pu = P_kW / (S_n_MVA * 1e3)
        return np.atleast_1d(p_ref_pu)
    
    def step_fmu(self, x, v, t, dt):
        P_e_pu  = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        P_out_kW = np.clip(P_e_pu * S_n_MVA * 1e3, 0.0, 1e8)
        GenSpeed_rpm = self.fmu.getReal([self.vrs['GenSpeed']])[0]

        #self.fmu.setReal([self.vrs['GenSpdOrTrq']], [GenSpeed_rpm])
        self.fmu.setReal([self.vrs['GenPwr']], [P_out_kW])
        self.fmu.setReal([self.vrs['ElecPwrCom']], [20e3])  # like DynaWind: MPPT

        self.fmu.doStep(currentCommunicationPoint=self._fmu_time,
                        communicationStepSize=self._fmu_dt)
        self._fmu_time += self._fmu_dt

    def terminate_fmu(self):
        self.fmu.terminate()
        self.fmu.freeInstance()