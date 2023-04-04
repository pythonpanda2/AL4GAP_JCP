# -*- coding: utf-8 -*-
"""
Created on Mon Oct 1 11:46:10 2021
    
@author: gsivaraman@anl.gov
"""


def run_quip(cutoff=5.0, delta=1.0, n_sparse=100, lmax=4, nmax=8):
    """
    A method to launch  quip and postprocess.
    Input args:
    cutoff : radial cutoff (float)
    delta: (0,1.0) (float)
    n_sparse : (int)
    lmax : (int)
    nmax : (int)
    return:
    [mae, rval] : (list)
    """
    import os
    import subprocess
    from ase.io import read
    from sklearn.metrics import mean_absolute_error, r2_score
    from time import time
    import numpy as np

    filename = 'species.dat'
    with open(filename, 'r') as f:
        species = f.readline().strip("\n")

    if os.path.isfile('quip_test.xyz'):
        os.remove('quip_test.xyz')
    if os.path.isfile('gap.xml'):
        os.remove('gap.xml')

    start = time()

    runtraining = " gap_fit at_file=./train.extxyz  gap={soap  cutoff=" + str(cutoff) + "  n_sparse=" + str(n_sparse) + "  covariance_type=dot_product sparse_method=cur_points  delta=" + str(delta) + "  zeta=4 l_max=" + str(
        lmax) + "  n_max=" + str(nmax) + "  atom_sigma=0.5  cutoff_transition_width=1.0  add_species} e0=" + str(species) + "  gp_file=gap.xml default_sigma={0.0001 0.0001 0.01 .01} sparse_jitter=1.0e-8 energy_parameter_name=energy "

    #print(runtraining)

    output = subprocess.check_output(
        runtraining, stderr=subprocess.STDOUT, shell=True)
    end = time()

    print("\nTraining Completed in {} s".format(np.round(end - start,2) ) )

    evaluate = "quip  E=T   atoms_filename=./test.extxyz   param_filename=gap.xml  | grep AT | sed 's/AT//' >> quip_test.xyz"
    pout = subprocess.check_output(
        evaluate, stderr=subprocess.STDOUT, shell=True)
    inp = read('test.extxyz', ':')
    inenergy = [ei.get_potential_energy() for ei in inp]
    output = read('quip_test.xyz', ':')
    outenergy = [eo.get_potential_energy() for eo in output]
    if len(inenergy) == len(outenergy):
        mae = mean_absolute_error(np.asarray(inenergy), np.asarray(outenergy))
        rval = r2_score(np.asarray(inenergy), np.asarray(outenergy))

        return [mae, rval]

    else:
        print("Memory blow up for cutoff={}, n_sparse={}".format(cutoff, n_sparse))
        return [float(20000), float(-2000)]