import os
import numpy as np
from src.dyn_models.utils import DAEModel
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


class FMUtoUICdrivetrain(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ 
        'FMUtoUICdrivetrain': {
            'FMUtoUICdrivetrain': [
                ['name', 'UIC', 'S_n', 'V_n', 'FMU_path', 'fmu_filename', 'control_mode', 'wd_path', 'openfast_test_dir',
                 'J_m', 'J_e', 'K', 'D', 'omega_m_rated', 'fmu_dt', 'ElecPwrCom_kW', 'efficiency'],
                ['FMUtoUICdrivetrain1', 'UIC1', 15, 22, 'FMU_path1', 'fmu_filename1', 3, 'wd_path1', 'openfast_test_dir1',
                 1.0e7, 1.0e6, 7.0e8, 7.0e7, 7.55, 0.01, 20000.0, 0.95756],
            ],
        }

        """
        # extract params and make sure the format is correct
        par = self.par
        sn = par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        par['S_n'] = sn
        self._efficiency = float(np.asarray(par['efficiency']).ravel()[0])
        self._fmu_dt = float(np.asarray(par['fmu_dt']).ravel()[0])
        omega_rated_rpm = float(np.asarray(par['omega_m_rated']).ravel()[0])
        J_m = float(np.asarray(par['J_m']).ravel()[0])
        J_e = float(np.asarray(par['J_e']).ravel()[0])
        K_SI = float(np.asarray(par['K']).ravel()[0])
        D_SI = float(np.asarray(par['D']).ravel()[0])

        # create useful conversion factors
        self._sys_to_local = self.sys_par['s_n'] / par['S_n']
        self._local_to_sys = par['S_n'] / self.sys_par['s_n']
        rpm_to_rad_s = 2.0 * np.pi / 60.0
        self._omega_base_rpm = omega_rated_rpm
        self._omega_base_rad_s = self._omega_base_rpm * rpm_to_rad_s
        self._T_base_Nm = sn * 1e6 / self._omega_base_rad_s

        # convert drivetrain params to pu
        # H_pu = 1/2 * J_SI * omega_base^2 / S_base
        self.H_m = 0.5 * J_m * self._omega_base_rad_s**2 / (sn * 1e6)
        self.H_e = 0.5 * J_e * self._omega_base_rad_s**2 / (sn * 1e6)
        #K_pu = K/T_base, D_pu = D*omega_base/T_base
        par['K'] = K_SI / self._T_base_Nm
        par['D'] = D_SI * self._omega_base_rad_s / self._T_base_Nm

        # Resolve FMU file path:
        # - If FMU_path already points to a .fmu, use it directly.
        # - Otherwise join FMU_path and fmu_filename.
        fmu_filename = str(np.atleast_1d(par['fmu_filename']).ravel()[0])
        fmu_path = str(np.atleast_1d(par['FMU_path']).ravel()[0])
        fmu_file = None
        if fmu_path and fmu_path.lower().endswith('.fmu'):
            fmu_file = fmu_path
        elif fmu_path and fmu_filename:
            fmu_file = os.path.join(fmu_path, fmu_filename)
        elif fmu_filename:
            # Backwards compatibility: allow callers that pass a full path as fmu_filename.
            fmu_file = fmu_filename
        else:
            raise KeyError("FMUtoUICdrivetrain requires 'FMU_path' and/or 'fmu_filename' to locate the .fmu.")

        if not os.path.isfile(fmu_file):
            raise FileNotFoundError(f"FMU file not found: {fmu_file}")

        model_description = read_model_description(fmu_file, validate=False)

        vrs = {}
        for variable in model_description.modelVariables:
            vrs[variable.name] = variable.valueReference

        print("Value References: \n")
        for name, vr in vrs.items():
            print(f"Variable: {name}, Value Reference: {vr}")

        unzipdir = extract(fmu_file)

        # OpenFAST fmu reads wd.txt from inside the extracted fmu resources folder
        # Writing to an arbitrary path in the repo (par['wd_path']) will not affect what the fmu reads
        new_directory = str(np.atleast_1d(par['openfast_test_dir']).ravel()[0])
        wd_file_path_in_fmu = os.path.join(unzipdir, 'resources', 'wd.txt')
        os.makedirs(os.path.dirname(wd_file_path_in_fmu), exist_ok=True)
        with open(wd_file_path_in_fmu, 'w') as f:
            f.write(new_directory)

        # also write to the user-provided path for visibility/debugging
        try:
            wd_file_path = str(np.atleast_1d(par['wd_path']).ravel()[0])
            if wd_file_path:
                os.makedirs(os.path.dirname(wd_file_path), exist_ok=True)
                with open(wd_file_path, 'w') as f:
                    f.write(new_directory)
        except Exception:
            pass

        fmu = FMU2Slave(guid=model_description.guid,
                        unzipDirectory=unzipdir,
                        modelIdentifier=model_description.coSimulation.modelIdentifier,
                        instanceName='instance1')

        if 'control_mode' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'control_mode'.")
        control_mode = int(np.atleast_1d(par['control_mode']).ravel()[0])

        if 'testNr' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'testNr'.")
        testNr = int(np.atleast_1d(par['testNr']).ravel()[0])

        fmu.instantiate()
        fmu.setReal([vrs['testNr']], [int(testNr)])
        fmu.setReal([vrs['Mode']], [int(control_mode)])

        print(f"[FMUtoUICdrivetrain] Using FMU: {fmu_file}", flush=True)
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()

        fmu.exitInitializationMode()

        self.fmu = fmu
        self.vrs = vrs
        if not np.isfinite(self._fmu_dt) or self._fmu_dt <= 0.0:
            raise ValueError(f"Invalid 'fmu_dt'={self._fmu_dt}. Must be a positive finite float (s).")
        self._last_fmu_comm_point = None
        self._fmu_warm_stepped = False
        # Cached fmu measurements (updated once per tops step in step_fmu)
        # Avoid reading fmu inside state_derivatives() since the solver may call it multiple times per step
        self._omega_m_pu_meas = None
        self._Te_pu_cmd = None
        self._gen_speed_rpm_meas = None
        self._gen_tq_kNm_meas = None
        self._gen_spdortrq_kNm_set = None
        self._genpwr_kW_set = None

        # Electrical power command to controller (kW)
        if 'ElecPwrCom_kW' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'ElecPwrCom_kW' (kW).")
        self._elec_pwr_com_kW = float(np.atleast_1d(par['ElecPwrCom_kW']).ravel()[0])
        if not np.isfinite(self._elec_pwr_com_kW) or self._elec_pwr_com_kW < 0.0:
            raise ValueError(
                f"Invalid 'ElecPwrCom_kW'={self._elec_pwr_com_kW}. Must be a finite float >= 0 (kW)."
            )

    def connections(self):
        # tops convention of input and output to connected model; here uic is connected
        # P_e is elec power output from uic, S_n_UIC is UIC base power: UIC -> FMUtoUICdrivetrain
        # P_ref is the power reference for the UIC: FMUtoUICdrivetrain -> UIC
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
        return ['omega_e', 'theta_s']

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def init_from_connections(self, x_0, v_0, S):
        self._input_values["P_e"] = self.P_e(x_0, v_0)
        self._input_values["S_n_UIC"] = self.S_n_UIC(x_0, v_0)

        # Initialize drivetrain states from fmu outputs
        X = self.local_view(x_0)
        par = self.par
        rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0]) # get initial rotor speed from fmu
        gen_rpm = float(self.fmu.getReal([self.vrs['GenSpeed']])[0]) # get initial generator speed from fmu
        self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0]) # get initial generator torque from fmu
        self._gen_speed_rpm_meas = gen_rpm # cache generator speed for next step
        self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm

        X['omega_e'] = gen_rpm / self._omega_base_rpm # initialize generator speed in pu

        # Electrical torque requested by coupling at init (local pu)
        Pe_uic_pu = float(np.asarray(self.P_e(x_0, v_0)).ravel()[0]) # get P_e output from UIC
        S_n_UIC = float(np.asarray(self.S_n_UIC(x_0, v_0)).ravel()[0]) # get UIC S_n base 
        Pe_pu = Pe_uic_pu * (S_n_UIC / par['S_n']) # convert P_e to local pu on S_n base
        Te_pu = Pe_pu / (self._efficiency * X['omega_e']) if abs(X['omega_e']) > 1e-6 else 0.0
        self._Te_pu_cmd = Te_pu # cache electrical torque command for next step

        X['theta_s'] = Te_pu / par['K'] # initialize shaft twist

        return

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        omega_m = float(self._omega_m_pu_meas) # from OpenFAST
        theta_s = float(np.asarray(X['theta_s']).ravel()[0]) # current tops state of shaft twist angle
        omega_e = float(np.asarray(X['omega_e']).ravel()[0]) # current tops state of generator speed

        # Electrical torque from power system:
        # P_e is in UIC p.u. on base S_n_UIC, convert to local p.u. on base S_n.
        Pe_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0]) # electrical power from UIC (pu on UIC base)
        S_n_UIC = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0]) # UIC base power
        Pe_pu = Pe_uic_pu * (S_n_UIC / par['S_n']) # UIC p P output on local base
        if not np.isfinite(omega_e):
            raise ValueError(f"FMUtoUICdrivetrain: omega_e is not finite (omega_e={omega_e}).")
        eff = float(self._efficiency)
        Te_pu = Pe_pu / (eff * omega_e) if abs(omega_e) > 1e-6 else 0.0 # output Pe is scaled with powertrain efficiency
        self._Te_pu_cmd = Te_pu # cache electrical torque command for next step

        # shaft torque
        omega_s = omega_m - omega_e
        K_pu = float(np.asarray(par['K']).ravel()[0])
        D_pu = float(np.asarray(par['D']).ravel()[0])
        T_shaft = (K_pu * theta_s + D_pu * omega_s) # shaft twist torque

        # swing eqs for drivetrain dynamics (pu)
        if not np.isfinite(T_shaft): # avoid simulations with NaN values
            raise ValueError(f"FMUtoUICdrivetrain: T_shaft is not finite (T_shaft={T_shaft}).")
        if not np.isfinite(Te_pu):
            raise ValueError(f"FMUtoUICdrivetrain: Te_pu is not finite (Te_pu={Te_pu}).")
        
        d_omega_e = (1 / (2.0 * self.H_e)) * (T_shaft - Te_pu)
        if not np.isfinite(d_omega_e): # avoid simulations with NaN values
            raise ValueError(
                "FMUtoUICdrivetrain: d(omega_e) is not finite "
                f"(d_omega_e={d_omega_e}, H_e={self.H_e}, T_shaft={T_shaft}, Te_pu={Te_pu})."
            )
        # tops update of state differentials
        dX['omega_e'] = d_omega_e
        dX['theta_s'] = omega_s
        
        return

    # FMU output names from modelDescription.xml (causality="output")
    FMU_OUTPUT_NAMES = [
        'Time', 'HSShftTq', 'GenTq', 'Wind1VelX', 'RtVAvgxh', 'BldPitch1',
        'NacYaw', 'RefGenSpd', 'GenSpeed', 'RotSpeed', 'LSSGagPxa', 'Azimuth',
        'GenAccel', 'YawBrTAxp', 'YawBrTAyp',
    ]

    def get_all_fmu_outputs(self):
        """Return dict of all FMU outputs (from modelDescription.xml)."""
        names = [n for n in self.FMU_OUTPUT_NAMES if n in self.vrs]
        if not names:
            return {}
        vrefs = [self.vrs[n] for n in names]
        vals = self.fmu.getReal(vrefs)
        return dict(zip(names, vals))

    def P_ref(self, x, v):
        # Use cached OpenFAST outputs to avoid repeated FMU reads during one solver step.
        # (The DAE solver may call output functions multiple times per time step.)
        X = self.local_view(x)
        omega_e = float(np.asarray(X['omega_e']).ravel()[0])

        P_kW  = (self._gen_tq_kNm_meas * omega_e * self._omega_base_rad_s)                        # since kN·m * rad/s / 1 = kW
        # Convert to pu on S_n 
        S_n_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        p_ref_pu = P_kW / (S_n_MVA * 1e3) # convert to pu on S_n base
        p_ref_pu = p_ref_pu * self._efficiency # scale with powertrain efficiency: Pgrid = Pgen * efficiency
        return np.atleast_1d(p_ref_pu)
    
    def step_fmu(self, x, v, t, dt):
        # NB! Because of a built-in function in fmu, any offset in the first 10 iterations will be compensated.
        # any disturbance should be applied after the first 10 iterations.
        par = self.par
        X = self.local_view(x)

        # Provide measured electrical power from the grid/UIC (kW on UIC base).
        P_e_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_uic_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        P_e_kW = P_e_uic_pu * S_n_uic_MVA * 1e3
        self._genpwr_kW_set = float(P_e_kW)
        self.fmu.setReal([self.vrs['GenPwr']], [float(P_e_kW)])

        # Torque command from accepted (x, v): Te = Pe/(eff*omega_e) in local pu.
        Pe_pu = P_e_uic_pu * (S_n_uic_MVA / par['S_n'])
        omega_e = float(np.asarray(X['omega_e']).ravel()[0])
        if not np.isfinite(omega_e):
            raise ValueError(f"FMUtoUICdrivetrain: omega_e is not finite (omega_e={omega_e}).")
        Te_pu = Pe_pu[0] / (self._efficiency * omega_e) if abs(omega_e) > 1e-6 else 0.0
        self._Te_pu_cmd = Te_pu

        # Send generator torque command (kN·m) into OpenFAST-FMU
        Te_kNm_cmd = (float(Te_pu)) * self._T_base_Nm / (1e3)
        input_torque = float(Te_kNm_cmd[0])
        if t > 5.0:
            input_torque = input_torque * 1.
        self.fmu.setReal([self.vrs['GenSpdOrTrq']], [input_torque])
        self._gen_spdortrq_kNm_set = input_torque
        
        # Demanded electrical power (kW) for controller.
        self._elec_pwr_com_kW_last = float(self._elec_pwr_com_kW)
        self.fmu.setReal([self.vrs['ElecPwrCom']], [float(self._elec_pwr_com_kW_last)])

        if abs(float(dt) - self._fmu_dt) > 1e-12:
            raise ValueError(f"FMUtoUICdrivetrain requires dt == fmu_dt. Got dt={dt}, fmu_dt={self._fmu_dt}.")
        comm_point = float(t) - float(dt)
        if comm_point < -1e-12:
            raise ValueError(f"Invalid FMU communication point t-dt={comm_point} (t={t}, dt={dt}).")
        # check if the FMU time is monotone -> t+1 > t
        if self._last_fmu_comm_point is not None and comm_point < self._last_fmu_comm_point - 1e-12:
            raise ValueError(f"Non-monotone FMU time: {comm_point} < last {self._last_fmu_comm_point}.")
        self._last_fmu_comm_point = comm_point

        self.fmu.doStep(currentCommunicationPoint=comm_point, communicationStepSize=dt)

        # Cache measurements for the next solver step
        rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
        self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
        self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
        self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])

        return

    def terminate_fmu(self):
        self.fmu.terminate()
        self.fmu.freeInstance()