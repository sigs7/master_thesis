"""
Minimal standalone test: run OpenFAST FMU exactly like DynaWind.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Like DynaWind: cwd = project root
os.chdir(project_root)

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave

if __name__ == '__main__':
    fmu_filename = 'fast.fmu'
    if not os.path.exists(fmu_filename):
        print(f"FMU not found: {fmu_filename}")
        sys.exit(1)

    # DynaWind: extract(fmu_filename) with NO unzipdir - uses temp dir
    unzipdir = extract(fmu_filename)
    new_directory = os.path.abspath('test1003').replace('/', '\\')
    wd_path = os.path.join(unzipdir, 'resources', 'wd.txt')
    if os.path.exists(os.path.dirname(wd_path)):
        with open(wd_path, 'w') as f:
            f.write(new_directory)

    model_description = read_model_description(fmu_filename, validate=False)
    vrs = {v.name: v.valueReference for v in model_description.modelVariables}

    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName='instance1',
    )
    fmu.instantiate()
    fmu.setReal([vrs['testNr']], [1003])
    fmu.setReal([vrs['Mode']], [3])
    fmu.setupExperiment(startTime=0.0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    print("Running FMU2Slave.doStep (like DynaWind step_fmu)...")
    dt = 0.01
    t = 0.0
    try:
        for _ in range(10):
            GenSpeed = fmu.getReal([vrs['GenSpeed']])[0]
            fmu.setReal([vrs['GenSpdOrTrq']], [GenSpeed])
            fmu.setReal([vrs['GenPwr']], [0.0])
            fmu.setReal([vrs['ElecPwrCom']], [20e3])
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
            t += dt
            GenTq, GenSpeed = fmu.getReal([vrs['GenTq'], vrs['GenSpeed']])
            print(f"  t={t:.2f} GenSpeed={GenSpeed:.1f} GenTq={GenTq:.1f}")
        fmu.terminate()
        fmu.freeInstance()
        print("SUCCESS!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
