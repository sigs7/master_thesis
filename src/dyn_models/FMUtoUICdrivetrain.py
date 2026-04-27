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
        par = self.par
        sn = par['S_n']
        sn[sn == 0] = self.sys_par['s_n']
        par['S_n'] = sn
        self._sys_to_local = self.sys_par['s_n'] / par['S_n']
        self._local_to_sys = par['S_n'] / self.sys_par['s_n']

        rpm_to_rad_s = 2.0 * np.pi / 60.0
        omega_rated_rpm = float(np.asarray(par['omega_m_rated']).ravel()[0])
        self._omega_base_rpm = omega_rated_rpm
        self._omega_base_rad_s = self._omega_base_rpm * rpm_to_rad_s

        S_n_MVA = float(np.asarray(par['S_n']).ravel()[0])
        self._S_base_W = S_n_MVA * 1e6
        self._T_base_Nm = self._S_base_W / self._omega_base_rad_s

        # Optional efficiency (match WindTurbine convention).
        # Interpreted as P_e = efficiency * P_mech  =>  T_mech = P_e / (efficiency * omega).
        
        self._efficiency = float(np.asarray(par['efficiency']).ravel()[0])

        J_m = float(np.asarray(par['J_m']).ravel()[0])
        J_e = float(np.asarray(par['J_e']).ravel()[0])
        self.H_m = 0.5 * J_m * self._omega_base_rad_s**2 / self._S_base_W
        self.H_e = 0.5 * J_e * self._omega_base_rad_s**2 / self._S_base_W

        # Convert drivetrain parameters to pu: K_pu = K/T_base, D_pu = D*omega_base/T_base
        K_SI = float(np.asarray(par['K']).ravel()[0])
        D_SI = float(np.asarray(par['D']).ravel()[0])
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

        # IMPORTANT: OpenFAST-FMU reads wd.txt from inside the extracted FMU resources folder.
        # Writing to an arbitrary path in the repo (par['wd_path']) will not affect what the FMU reads.
        new_directory = str(np.atleast_1d(par['openfast_test_dir']).ravel()[0])
        wd_file_path_in_fmu = os.path.join(unzipdir, 'resources', 'wd.txt')
        os.makedirs(os.path.dirname(wd_file_path_in_fmu), exist_ok=True)
        with open(wd_file_path_in_fmu, 'w') as f:
            f.write(new_directory)

        # Optional: also write to the user-provided path for visibility/debugging.
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
        fmu.instantiate()
        fmu.setReal([vrs['testNr']], [1002]) 
        if 'control_mode' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'control_mode'.")
        control_mode = int(np.atleast_1d(par['control_mode']).ravel()[0])
        fmu.setReal([vrs['Mode']], [control_mode])


        print(f"[FMUtoUICdrivetrain] Using FMU: {fmu_file}", flush=True)
        fmu.setupExperiment(startTime=0.0)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

        self.fmu = fmu
        self.vrs = vrs
        if 'fmu_dt' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'fmu_dt' (s).")
        self._fmu_dt = float(np.atleast_1d(par['fmu_dt']).ravel()[0])
        if not np.isfinite(self._fmu_dt) or self._fmu_dt <= 0.0:
            raise ValueError(f"Invalid 'fmu_dt'={self._fmu_dt}. Must be a positive finite float (s).")
        self._last_fmu_comm_point = None
        self._fmu_warm_stepped = False
        # Cached OpenFAST measurements (updated once per TOPS step in step_fmu).
        # Avoid reading FMU inside state_derivatives() since the solver may call it multiple times per step.
        self._omega_m_pu_meas = None
        self._Te_pu_cmd = None
        self._gen_speed_rpm_meas = None
        self._gen_tq_kNm_meas = None
        # Debug/logging: last value written to FMU input GenSpdOrTrq (rpm).
        # Note: in the OpenFAST-FMU "Mode=3" coupling used here, this input is treated as
        # generator speed feedback (see standalone test in casestudies/dyn_sim/test_fmu_standalone.py).
        self._gen_spdortrq_rpm_set = None
        # Optional debug scaling of GenSpdOrTrq input to verify write-through.
        # Set env var FMU_GENSPDORTRQ_SCALE (e.g. "0.5" or "2.0") to apply scaling.
        try:
            self._gen_spdortrq_scale = float(os.getenv('FMU_GENSPDORTRQ_SCALE', '').strip() or '1.0')
        except Exception:
            self._gen_spdortrq_scale = 1.0
        # Electrical power command to controller (kW)
        if 'ElecPwrCom_kW' not in par.dtype.names:
            raise KeyError("FMUtoUICdrivetrain requires parameter 'ElecPwrCom_kW' (kW).")
        self._elec_pwr_com_kW = float(np.atleast_1d(par['ElecPwrCom_kW']).ravel()[0])
        if not np.isfinite(self._elec_pwr_com_kW) or self._elec_pwr_com_kW < 0.0:
            raise ValueError(
                f"Invalid 'ElecPwrCom_kW'={self._elec_pwr_com_kW}. Must be a finite float >= 0 (kW)."
            )

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
        # omega_m is provided by OpenFAST (FMU) each step, not integrated in TOPS.
        return ['omega_e', 'theta_s']

    def input_list(self):
        return ['P_e', 'S_n_UIC'] 

    def output_list(self):
        return ['P_ref']
    
    def init_from_connections(self, x_0, v_0, S):
        self._input_values["P_e"] = self.P_e(x_0, v_0)
        self._input_values["S_n_UIC"] = self.S_n_UIC(x_0, v_0)

        # Initialize drivetrain states from OpenFAST outputs.
        X = self.local_view(x_0)
        rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
        gen_rpm = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
        if 'GenTq' in self.vrs:
            self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])
        self._gen_speed_rpm_meas = gen_rpm

        eps_rpm = 1e-3
        if abs(rot_rpm) < eps_rpm and abs(gen_rpm) < eps_rpm:
            rot_rpm = self._omega_base_rpm
            gen_rpm = self._omega_base_rpm
        elif abs(rot_rpm) < eps_rpm and abs(gen_rpm) >= eps_rpm:
            rot_rpm = gen_rpm
        elif abs(gen_rpm) < eps_rpm and abs(rot_rpm) >= eps_rpm:
            gen_rpm = rot_rpm
        else:
            # Direct-drive assumption in this simplified coupling:
            # avoid an artificial initial slip (omega_m - omega_e) that creates a torque kick in omega_e.
            if abs(rot_rpm - gen_rpm) > eps_rpm:
                rot_rpm = gen_rpm

        X['omega_e'] = gen_rpm / self._omega_base_rpm
        # Cache omega_m for the first solver call (avoids initial 0->1 pu jump if FMU outputs lag at t=0).
        self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
        # Also seed generator-speed cache with the adjusted init value used by omega_e.
        self._gen_speed_rpm_meas = gen_rpm

        # Initialize shaft twist.
        # At t=0 the FMU can report HSShftTq=0 for a step or two; if we use that directly,
        # we start with T_shaft≈0 while the grid-side coupling immediately requests Te=Pe/omega_e,
        # which creates a large transient. Prefer a consistent torque-balance init:
        # assume omega_s≈0 and set theta_s so that T_shaft≈Te (or use HSShftTq if it's available/nonzero).
        K = float(np.asarray(self.par['K']).ravel()[0])
        if abs(K) <= 1e-12:
            X['theta_s'] = 0.0
            return

        # Electrical torque requested by coupling at init (local pu)
        Pe_uic_pu = float(np.asarray(self.P_e(x_0, v_0)).ravel()[0])
        S_n_UIC = float(np.asarray(self.S_n_UIC(x_0, v_0)).ravel()[0])
        S_n_WT = float(np.asarray(self.par['S_n']).ravel()[0])
        Pe_pu = Pe_uic_pu * (S_n_UIC / S_n_WT)
        omega_e = float(np.asarray(X['omega_e']).ravel()[0])
        eff = float(self._efficiency)
        Te_pu = Pe_pu / (eff * omega_e) if abs(omega_e) > 1e-6 else 0.0
        self._Te_pu_cmd = Te_pu

        X['theta_s'] = Te_pu / K

        # Warm-start: OpenFAST-FMU outputs can stay at defaults at t=0 until the first doStep().
        # Push initial inputs and do one warm-up step 0->dt here. Then we skip the first doStep()
        # in step_fmu() to keep alignment with TOPS time steps.
        try:
            import time as _time
            t0 = _time.perf_counter()
            print(f"[FMUtoUICdrivetrain] warm-start begin (dt={float(self._fmu_dt):g}s)", flush=True)
            if 'GenPwr' in self.vrs:
                Pe_uic_pu = float(np.asarray(self.P_e(x_0, v_0)).ravel()[0])
                S_n_UIC = float(np.asarray(self.S_n_UIC(x_0, v_0)).ravel()[0])
                P_e_kW = Pe_uic_pu * float(S_n_UIC) * 1e3
                self.fmu.setReal([self.vrs['GenPwr']], [P_e_kW])
            if 'GenSpdOrTrq' in self.vrs:
                omega_e_pu_0 = float(np.asarray(X['omega_e']).ravel()[0])
                gen_rpm_in = omega_e_pu_0 * float(self._omega_base_rpm)
                gen_rpm_in = float(gen_rpm_in) * float(self._gen_spdortrq_scale)
                self._gen_spdortrq_rpm_set = float(gen_rpm_in)
                self.fmu.setReal([self.vrs['GenSpdOrTrq']], [float(gen_rpm_in)])
            if 'ElecPwrCom' in self.vrs:
                self.fmu.setReal([self.vrs['ElecPwrCom']], [float(self._elec_pwr_com_kW)])

            dt_ws = float(self._fmu_dt)
            print("[FMUtoUICdrivetrain] calling fmu.doStep(0->dt)...", flush=True)
            self.fmu.doStep(currentCommunicationPoint=0.0, communicationStepSize=dt_ws)
            print(f"[FMUtoUICdrivetrain] fmu.doStep returned in {(_time.perf_counter()-t0):.2f}s", flush=True)
            self._fmu_warm_stepped = True
            self._last_fmu_comm_point = 0.0

            # Refresh caches after warm-start (FMU is now at t=dt).
            if 'RotSpeed' in self.vrs:
                rot_rpm_ws = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
                if abs(rot_rpm_ws) >= eps_rpm:
                    self._omega_m_pu_meas = rot_rpm_ws / self._omega_base_rpm
            if 'GenSpeed' in self.vrs:
                gen_rpm_ws = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
                if abs(gen_rpm_ws) >= eps_rpm:
                    self._gen_speed_rpm_meas = gen_rpm_ws
                    # Keep initial TOPS state consistent with the post-warmstart FMU speed.
                    # Otherwise we start with a slip/torque mismatch that creates a spike in omega_e.
                    X['omega_e'] = gen_rpm_ws / self._omega_base_rpm
                    # Direct-drive assumption in this simplified coupling: enforce omega_m ≈ omega_e at init.
                    self._omega_m_pu_meas = float(np.asarray(X['omega_e']).ravel()[0])
            if 'GenTq' in self.vrs:
                self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])
        except Exception:
            # If warm-start fails, keep existing init logic.
            pass

        print(f"K: {K}, D: {self.par['D'][0]}, He: {self.H_e}, Hm: {self.H_m}")

        return

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        omega_m = float(self._omega_m_pu_meas)
        theta_s = float(np.asarray(X['theta_s']).ravel()[0])
        omega_e = float(np.asarray(X['omega_e']).ravel()[0])

        # Electrical torque from power system:
        # P_e is in UIC p.u. on base S_n_UIC, convert to local p.u. on base S_n.
        Pe_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_UIC = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        S_n_loc = float(np.asarray(self.par['S_n']).ravel()[0])
        Pe_pu = Pe_uic_pu * (S_n_UIC / S_n_loc) if S_n_loc > 0 else Pe_uic_pu
        if not np.isfinite(omega_e):
            raise ValueError(f"FMUtoUICdrivetrain: omega_e is not finite (omega_e={omega_e}).")
        if abs(omega_e) < 1e-6:
            raise ValueError(
                f"FMUtoUICdrivetrain: omega_e too small for Te=Pe/omega_e "
                f"(omega_e={omega_e}, Pe_pu(local)={Pe_pu})."
            )
        eff = float(self._efficiency)
        Te_pu = Pe_pu / (eff * omega_e)
        self._Te_pu_cmd = Te_pu
        # shaft torque
        omega_s = omega_m - omega_e
        K_pu = float(np.asarray(par['K']).ravel()[0])
        D_pu = float(np.asarray(par['D']).ravel()[0])
        T_shaft = (K_pu * theta_s + D_pu * omega_s)

        # swing eqs for drivetrain dynamics (pu)
        # Use H_m/H_e computed from (J_m/J_e) once bases are known.
        H_e = float(self.H_e)
        if not np.isfinite(T_shaft):
            raise ValueError(f"FMUtoUICdrivetrain: T_shaft is not finite (T_shaft={T_shaft}).")
        if not np.isfinite(Te_pu):
            raise ValueError(f"FMUtoUICdrivetrain: Te_pu is not finite (Te_pu={Te_pu}).")
        d_omega_e = (1 / (2.0*H_e)) * (T_shaft - Te_pu)
        if not np.isfinite(d_omega_e):
            raise ValueError(
                "FMUtoUICdrivetrain: d(omega_e) is not finite "
                f"(d_omega_e={d_omega_e}, H_e={H_e}, T_shaft={T_shaft}, Te_pu={Te_pu})."
            )
        dX['omega_e'] = d_omega_e
        dX['theta_s'] = omega_s
        
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
        # Use cached OpenFAST outputs to avoid repeated FMU reads during one solver step.
        # (The DAE solver may call output functions multiple times per time step.)
        if self._gen_tq_kNm_meas is None or self._gen_speed_rpm_meas is None:
            GenTq_kNm, GenSpeed_rpm = self.fmu.getReal([self.vrs['GenTq'], self.vrs['GenSpeed']])
            self._gen_tq_kNm_meas = float(GenTq_kNm)
            self._gen_speed_rpm_meas = float(GenSpeed_rpm)
        GenTq_kNm = float(self._gen_tq_kNm_meas)
        GenSpeed_rpm = float(self._gen_speed_rpm_meas)
        # Convert units
        omega = GenSpeed_rpm * 2.0 * np.pi / 60.0              # rad/s
        P_kW  = (GenTq_kNm * omega)                            # since kN·m * rad/s / 1 = kW
        # Convert to pu on S_n 
        S_n_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        p_ref_pu = P_kW / (S_n_MVA * 1e3)
        eff = float(self._efficiency)
        p_ref_pu = p_ref_pu * eff
        return np.atleast_1d(p_ref_pu)
    
    def step_fmu(self, x, v, t, dt):
        if 'GenPwr' not in self.vrs:
            raise KeyError("FMUtoUICdrivetrain requires FMU input 'GenPwr'.")
        if 'ElecPwrCom' not in self.vrs:
            raise KeyError("FMUtoUICdrivetrain requires FMU input 'ElecPwrCom'.")
        if 'GenSpdOrTrq' not in self.vrs:
            raise KeyError("FMUtoUICdrivetrain requires FMU input 'GenSpdOrTrq'.")

        # Provide measured electrical power from the grid/UIC (kW on UIC base).
        P_e_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_uic_MVA = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        P_e_kW = P_e_uic_pu * S_n_uic_MVA * 1e3
        self.fmu.setReal([self.vrs['GenPwr']], [P_e_kW])

        # Torque command from accepted (x, v): Te = Pe/(eff*omega_e) in local pu.
        X = self.local_view(x)
        Pe_uic_pu = float(np.asarray(self.P_e(x, v)).ravel()[0])
        S_n_UIC = float(np.asarray(self.S_n_UIC(x, v)).ravel()[0])
        S_n_loc = float(np.asarray(self.par['S_n']).ravel()[0])
        Pe_pu = Pe_uic_pu * (S_n_UIC / S_n_loc) if S_n_loc > 0 else Pe_uic_pu
        omega_e = float(np.asarray(X['omega_e']).ravel()[0])
        if not np.isfinite(omega_e):
            raise ValueError(f"FMUtoUICdrivetrain: omega_e is not finite (omega_e={omega_e}).")
        if abs(omega_e) < 1e-6:
            raise ValueError(
                f"FMUtoUICdrivetrain: omega_e too small for Te=Pe/omega_e "
                f"(omega_e={omega_e}, Pe_pu(local)={Pe_pu})."
            )
        eff = float(self._efficiency)
        Te_pu = Pe_pu / (eff * omega_e)
        self._Te_pu_cmd = Te_pu
        # Feed generator speed (rpm) into OpenFAST-FMU.
        gen_rpm_in = float(omega_e * float(self._omega_base_rpm)) * float(self._gen_spdortrq_scale)
        self._gen_spdortrq_rpm_set = float(gen_rpm_in)
        self.fmu.setReal([self.vrs['GenSpdOrTrq']], [float(gen_rpm_in)])

        # Demanded electrical power (kW) for controller.
        # Note: FMU variable is in kW (not W).
        self._elec_pwr_com_kW_last = float(self._elec_pwr_com_kW)
        self.fmu.setReal([self.vrs['ElecPwrCom']], [self._elec_pwr_com_kW_last])

        # If we advanced the FMU once during init_from_connections (0->dt), skip the
        # first doStep() call in the simulation loop to keep alignment.
        if self._fmu_warm_stepped:
            self._fmu_warm_stepped = False
            # Cache measurements (FMU already at t=dt).
            if 'RotSpeed' in self.vrs:
                rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
                self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
            if 'GenSpeed' in self.vrs:
                self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
            if 'GenTq' in self.vrs:
                self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])
            return

        dt = float(dt)
        t = float(t)
        if abs(dt - self._fmu_dt) > 1e-12:
            raise ValueError(f"FMUtoUICdrivetrain requires dt == fmu_dt. Got dt={dt}, fmu_dt={self._fmu_dt}.")
        comm_point = t - dt
        if comm_point < -1e-12:
            raise ValueError(f"Invalid FMU communication point t-dt={comm_point} (t={t}, dt={dt}).")
        # Optional monotonicity check
        if self._last_fmu_comm_point is not None and comm_point < self._last_fmu_comm_point - 1e-12:
            raise ValueError(f"Non-monotone FMU time: {comm_point} < last {self._last_fmu_comm_point}.")
        self._last_fmu_comm_point = comm_point

        self.fmu.doStep(currentCommunicationPoint=comm_point, communicationStepSize=dt)

        # Cache measurements for the next solver step.
        if 'RotSpeed' in self.vrs:
            rot_rpm = float(self.fmu.getReal([self.vrs['RotSpeed']])[0])
            self._omega_m_pu_meas = rot_rpm / self._omega_base_rpm
        if 'GenSpeed' in self.vrs:
            self._gen_speed_rpm_meas = float(self.fmu.getReal([self.vrs['GenSpeed']])[0])
        if 'GenTq' in self.vrs:
            self._gen_tq_kNm_meas = float(self.fmu.getReal([self.vrs['GenTq']])[0])

    def terminate_fmu(self):
        self.fmu.terminate()
        self.fmu.freeInstance()