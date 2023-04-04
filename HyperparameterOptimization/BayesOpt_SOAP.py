"""
@author: gsivaraman@anl.gov
"""
import GPyOpt
import json
import os
import numpy as np
from ase.io import read, write
import subprocess
from pprint import pprint
from time import time
from sklearn.metrics import mean_absolute_error, r2_score


def run_quip(cutoff=4.5, delta2b=.001, delta=1.0, n_sparse=100, n_sparse2b=50, lmax=4, nmax=8):
    """
    Descriptor specific input
    """
    if os.path.isfile('quip_test.xyz'):
        os.remove('quip_test.xyz')
    if os.path.isfile('gap.xml'):
        os.remove('gap.xml')


    start = time()
    quippath = "/home/gsivaraman/libatoms/QUIP/build/linux_x86_64_ifort_icc_openmp/"
    commands = quippath+"gap_fit  at_file=train.xyz  gap={distance_2b  cutoff="+str(cutoff)+"  n_sparse="+str(n_sparse2b)+"  sparse_method=uniform  delta="+str(delta2b)+"  covariance_type=ard_se theta_uniform=1.0  cutoff_transition_width=1.0  add_species : soap  cutoff="+str(cutoff)+"  n_sparse="+str(
        n_sparse)+"  covariance_type=dot_product sparse_method=cur_points  delta="+str(delta)+"  zeta=4 l_max="+str(lmax)+"  n_max="+str(nmax)+"  atom_sigma=0.5  cutoff_transition_width=1.0  add_species}   gp_file=gap.xml   default_sigma={0.001 0.1 0.01 0.0} sparse_jitter=1.0e-8 energy_parameter_name=energy "  # force_parameter_name=force  virial_parameter_name =virial
    # print(screenout)
    output = subprocess.check_output(commands,stderr=subprocess.STDOUT, shell=True)
    end = time()
    print("\nTraining Completed in {} sec".format(end - start))

    testoutstat = subprocess.check_output(quippath +
                                                   "quip  E=T   atoms_filename=test.xyz   param_filename=gap.xml  | grep AT | sed 's/AT//' >> quip_test.xyz",stderr=subprocess.STDOUT, shell=True)

    inp = read('test.xyz', ':')
    inenergy = [ei.get_potential_energy() for ei in inp]
    output = read('quip_test.xyz', ':')
    outenergy = [eo.get_potential_energy() for eo in output]
    if len(inenergy) == len(outenergy):
        mae = mean_absolute_error(np.asarray(outenergy), np.asarray(inenergy))
        rscore = r2_score(np.asarray(outenergy), np.asarray(inenergy))
        return [mae, rscore]

    else:
        print("Memory blow up for cutoff={}, n_sparse={}".format(cutoff, n_sparse))
        return [float(20000), float(20000)]


bounds = [{'name': 'cutoff',             'type': 'continuous', 'domain': (4, 8)},
          {'name': 'delta2b',            'type': 'continuous',
              'domain': (1, 20.0)},
          {'name': 'delta',            'type': 'continuous',
              'domain': (0.1, 0.99)},
          {'name': 'n_sparse',        'type': 'discrete',
              'domain': range(100, 1501, 100)},
          {'name': 'n_sparse2b',        'type': 'discrete',
              'domain': range(10, 105, 5)},
          {'name': 'lmax',            'type': 'discrete',  'domain': (4, 5, 6)},
          {'name': 'nmax',            'type': 'discrete',  'domain': (8, 9, 10, 11, 12)}, ]


# function to optimize MP model
def f(x):
  
    evaluation = run_quip(cutoff=float(x[:, 0]), delta2b=float(x[:, 1]), delta=float(x[:, 2]), n_sparse=float(
        x[:, 3]), n_sparse2b=float(x[:, 4]), lmax=float(x[:, 5]), nmax=float(x[:, 6]) )
    print("\nParam: {}  |  MAE, R2: {}".format(x, evaluation))

    return evaluation[0]


# optimizer
opt_quip = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=35,
                                               model_type="GP_MCMC",
                                               acquisition_type='EI_MCMC',  # EI
                                               evaluator_type="predictive",  # Expected Improvement
                                               exact_feval=False)

# optimize MP model
print("\nBegin Optimization run \t")

opt_quip.run_optimization(max_iter=75)
x_best = opt_quip.x_opt


print("\n\n Best parameters :\n")
print(x_best)

print("\nOptimized Parameters:\n")


hyperdict = {}
for num in range(len(bounds)):
    hyperdict[bounds[num]["name"]] = opt_quip.x_opt[num]

hyperdict["MAE"] = opt_quip.fx_opt

print("\n Writing output : hyperparam_quip.json\n")

with open('hyperparam_quip.json', 'w') as outfile:
    json.dump(hyperdict, outfile)

print("\n Printing best hyper parameters : \n")

pprint(hyperdict)
