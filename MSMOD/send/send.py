# -*- coding: utf-8 -*-
"""    
@author:  Vanessa Woo
"""
from AL4GAP.AL4GAP import run_AL
from AL4GAP.parse_trj import parse_lammps
import argparse
import os
import time

parser = argparse.ArgumentParser(description="Path to LAMMPS_ensemble run")
parser.add_argument("-p", "--path", type=str)
args = parser.parse_args()
runpath = args.path

def parse(runpath):
        os.chdir(runpath)
        parse_lammps(runpath)
        return

def AL(runpath):
        os.chdir(runpath)
        run_AL(runpath,
                xyzfilename='To_QUIP.extxyz',
                nsample=10,
                nminclust=30,
                cutoff=(4, 7),
                sparse=(100, 1200),
                lmax=(4, 6),
                nmax=(7, 12),
                Nopt=(10, 20),
                Etol=0.09)
        
def file_exists(runpath):
        final_config = f"{runpath}final_config.data"
        while not os.path.exists(final_config):
                time.sleep(60*5)
                print(f"waiting for: {final_config}", flush=True)
        else:
                time.sleep(60)
                parse(runpath)
                AL(runpath)

file_exists(runpath)
