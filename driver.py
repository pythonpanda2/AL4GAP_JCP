# -*- coding: utf-8 -*-
"""    
@author:  Vanessa Woo
"""
import os 
import sys
import pandas as pd 
from smartsim import Experiment 
from smartredis import Client
from AL4GAP.setup_inputs import setup_inputs


# setup the LAMMPS input file from the experimental density file (currently 3 compositions)
df = pd.read_csv('densities.csv')
runpath, lmpFilelist = setup_inputs(df,ffparam='OPLS')  # returns a list of run paths for LAMMPS simulation
num_exp = len(runpath)  # number of compositions = number of experiments to run
print(f"Number of experiments {num_exp}")
base_path = os.path.abspath(os.curdir)          
CONDA_SH = "/home/ac.vwoo/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV = "/home/ac.vwoo/miniconda3/envs/ss-al4gap"
PORT = 6780

class AL4GAP:

    # Create SmartSim experiment
    def __init__(self):
        # All AL4GAP output will be generated in "Output" directory
        self.exp = Experiment("Output", launcher="slurm")
        self.exp.generate(overwrite=True)

    # Create clustered orchestrator 
    def create_orchestrator(self):
        self.orchestrator = self.exp.create_database(db_nodes=dbnodes,
                                                    port=PORT,
                                                    interface='lo',
                                                    time='10:00:00')
        self.exp.generate(self.orchestrator)
        self.exp.start(self.orchestrator)
        self.client = Client(address=self.orchestrator.get_address()[0])
        return

    def LAMMPS_ensemble(self, num_exp):
        """
        Generate LAMMPS_ensemble for given MS compositions
        "model" instances are used to run AL4GAP tasks
        :param num_exp: number of input compositions
        :type num_exp: int
        """
        batch_args = {'account':'AL-IP',
                      'partition':'bdwall',
                      'exclusive':None
                                        }
        # BEBOP modules for LAMMPS executable
        ml = "intel/cluster.2018.3 gcc/7.1.0 gsl/2.4 lammps/12Mar19"
        batch_settings = self.exp.create_batch_settings(nodes=simnodes,
                                                        time='8:00:00',
                                                        batch_args=batch_args)
        # add preambles to LAMMPS_ensemble batch script 
        batch_settings.add_preamble(f'module load {ml}')
        batch_settings.add_preamble('export OMP_NUM_THREADS=1')
        LAMMPS_ensemble = self.exp.create_ensemble("LAMMPS_ensemble", batch_settings=batch_settings)
        
        # ---------------------------------------------------------
        # Run LAMMPS simluation
        lmp_run_args = {'nodes':node,'ntasks':ntasks}
        lmp_exe = "/soft/lammps/12Mar19/lmp_intel_cpu_intelmpi_bdw"
        lmp_env_vars = {"LD_LIBRARY_PATH":f"{lmp_exe}:$LD_LIBRARY_PATH", 
                        "OMP_NUM_THREADS":1, "exclusive":None}
        for i in range(num_exp):
            lmp_input=f"{base_path}/Data/{i+1}/opls.in"
            lmp_exe_args = f' -in {lmp_input}'
            lmp_run_settings = self.exp.create_run_settings(run_command='srun',
                                                            run_args=lmp_run_args, 
                                                            exe=lmp_exe, 
                                                            exe_args=lmp_exe_args, 
                                                            env_vars=lmp_env_vars)
            lmp_model = self.exp.create_model(f'LAMMPS_{i+1}', run_settings=lmp_run_settings)
            lmp_model.attach_generator_files([f"{base_path}/Data/1/species.dat"])
            LAMMPS_ensemble.add_model(lmp_model)
        # Generate LAMMPS_ensemble batch file
        self.exp.generate(LAMMPS_ensemble, overwrite=True)
        return LAMMPS_ensemble
    
    def AL_ensemble(self, num_exp):
        # ---------------------------------------------------------
        # Parse LAMMPS trajectories and run AL
        batch_args = {'account':'AL-IP',
                      'partition':'bdwall',
                      'exclusive':None
                                            }
        batch_settings = self.exp.create_batch_settings(nodes=node,
                                                        time='10:00:00',
                                                        batch_args=batch_args)
        batch_settings.add_preamble(f"source {CONDA_SH}")
        batch_settings.add_preamble(f"conda activate {CONDA_ENV}")
        AL_ensemble = self.exp.create_ensemble("AL_ensemble", batch_settings=batch_settings)
        # set path variable 
        path = os.getenv("PATH", "")
        python_path = f"{base_path}:{base_path}/MSMOD:{base_path}/MSMOD/AL4GAP:" + path
        os.environ["PATH"]=python_path
        AL_env_vars = {"PATH": python_path}
        for i in range(num_exp):
            lmp_path = f"{base_path}/Output/LAMMPS_ensemble/LAMMPS_{i+1}/"
            AL_exe_args = [f"{base_path}/MSMOD/send/send.py", f"--path={lmp_path}"]
            AL_run_settings = self.exp.create_run_settings(run_command='auto',
                                                            exe='python', 
                                                            exe_args= AL_exe_args, 
                                                            env_vars=AL_env_vars
                                                            )
            AL_model = self.exp.create_model(f"AL_{i+1}", run_settings=AL_run_settings)
            AL_ensemble.add_model(AL_model)
        # Generate AL4GAP_ensemble batch file
        self.exp.generate(AL_ensemble, overwrite=True)
        return AL_ensemble

    def run_AL4GAP(self):
        """
        Run Orchestrator and AL4GAP pipeline
        """
        self.create_orchestrator()
        self.LAMMPS = self.LAMMPS_ensemble(num_exp)
        self.AL = self.AL_ensemble(num_exp)
        # both ensembles are launched, but AL_ensemble remains idle until LAMMPS simulation is completed
        self.exp.start(self.LAMMPS, block=False, summary=True)
        self.exp.start(self.AL, block=False, summary=True)             

if __name__ == '__main__':

    # Parse command line arguments
    node = int(sys.argv[1])
    dbnodes = int(sys.argv[2])
    simnodes = int(sys.argv[3])
    ntasks = int(sys.argv[4])
    hostfile = sys.argv[5]
    jobid = sys.argv[6]

    pipeline = AL4GAP()
    pipeline.run_AL4GAP()
