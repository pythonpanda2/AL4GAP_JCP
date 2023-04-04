# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 11:01:10 2021
    
@author: gsivaraman@anl.gov
"""

from __future__ import print_function


def gen_trials(minclustlen, maxclustlen):
    '''
    Generate the trials in the   sampling width (Kmax, Kmin)
    :param minclustlen: Kmin (int)
    :param maxclustlen: Kmax (int)
    Return type: Trials  (list)
    '''
    import numpy as np
    if maxclustlen <= 50:
        # int( np.floor(data.nsample/2) )
        trialscale = int(np.floor(minclustlen/2))
        maxtrial = int(np.floor(maxclustlen/trialscale))
        trials = [trialscale*t for t in range(maxtrial, 0, -1)]
    elif maxclustlen > 50 and maxclustlen <= 200:
        trialscale = int(np.floor((minclustlen/2) * (minclustlen/2)))
        maxtrial = int(np.floor(maxclustlen/trialscale))
        trials = [trialscale*t for t in range(maxtrial, 0, -1)] + [u*int(np.floor(
            minclustlen/2)) for u in range(int(np.floor(minclustlen/2)) - 1, 0, -1)]
    elif maxclustlen > 200:
        trialscale = int(minclustlen**2)
        maxtrial = int(np.floor(maxclustlen/trialscale))
        trials = [trialscale*t for t in range(maxtrial, 0, -1)] + [u*int(
            np.floor(minclustlen/2)) for u in range(int(minclustlen), 0, -1)]
    return trials


def f(x):
    """
    Surrogate function over the error metric to be optimized
    """
    from .launch_quip import run_quip
    import numpy as np     

    evaluation = run_quip(cutoff=float(x[:, 0]), delta=float(
        x[:, 1]), n_sparse=int(x[:, 2]), lmax=int(x[:, 3]), nmax=int(x[:, 4]))

    print("\nParam: Cutoff= {}, delta= {}, n_sparse= {}, lmax= {}, nmax= {}  |  MAE : {} eV, R2: {}".format( np.round( float(x[:, 0]),2), np.round(
        float(x[:, 1]),2), int(x[:, 2]), int(x[:, 3]), int(x[:, 4]), np.round(evaluation[0],2), np.round(evaluation[1],2) ) )

    return evaluation[0]


def plot_metric(track_metric):
    '''
    Plot the metric evolution over the trial using this function
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.style.use('seaborn')
    cmap = plt.cm.get_cmap('brg')
    plt.rcParams["font.family"] = "sans-serif"
    x = range(1, len(track_metric)+1)
    plt.plot(x, track_metric, 'ro', c=cmap(0.40), linewidth=3.0)
    plt.title('Active Learning', fontsize=32)
    plt.xlabel('Trial', fontsize=24)
    plt.ylabel('MAE (eV)', fontsize=24)
    plt.grid('True')
    plt.tight_layout()
    plt.draw()
    plt.savefig('AL.png', dpi=300)
    plt.savefig('AL.svg', dpi=300, format='svg')
    c = 'Done'
    return c


def run_AL(execpath,
           xyzfilename='To_QUIP.extxyz',
           nsample=10,
           nminclust=30,
           cutoff=(4, 7),
           sparse=(100, 1200),
           lmax=(4, 6),
           nmax=(7, 12),
           Nopt=(10, 20),
           Etol=0.09):
    '''
    A wrap around function to invoke on-the-fly AL and BO. 
    :param execpath: str
    :param xyzfilename:  str
    :param nsample:  int
    :param nminclust : int
    :param cutoff : float tuple
    :param sparse : int tuple
    :param lmax : int tuple
    :param nmax : int tuple
    :param Nopt : int tuple
    :param Etol : float
    Return type: None
    '''

    import GPyOpt
    import json
    import os
    import random
    import subprocess
    import numpy as np
    from tqdm import tqdm
    from ase.io import read, write
    from copy import deepcopy
    from .activesample import activesample
    from hdbscan import HDBSCAN
    import mdtraj as md
    from copy import deepcopy
    from .AL4GAP import gen_trials, f, plot_metric

    # cutoff = tuple(args.cutoff)
    # sparse = tuple(args.nsparse)
    # nlmax = tuple(args.nlmax)
    # Nopt = tuple(args.Nopt)
    # Etol = float(args.precision)
    os.chdir(execpath)
    trackerjson = {}
    track_metric = []  # For plotting the error metric
    trackerjson['clusters'] = []

    if 'deform' in execpath:
        nminclust = 25

    data = activesample(
        xyzfilename, nsample=nsample, nminclust=nminclust)
    Nclust, Nnoise, clust = data.gen_cluster()

    if Nclust < 10:
        data.nminclust = 10
        Nclust, Nnoise, clust = data.gen_cluster()

    clustlen = []

    for Ni, Ntraj in clust.items():
        clustlen.append(len(Ntraj))

    maxclustlen = max(clustlen)
    minclustlen = min(clustlen)

    print("\nNumber of elements in the smallest, largest cluster is {}, {}\n".format(
        minclustlen, maxclustlen))
    print("\n Nnoise : {}, Nclusters : {}\n".format(Nnoise, Nclust))

    trackerjson['Nclusters'] = Nclust
    trackerjson['Nnoise'] = Nnoise
    trackerjson['clusters'].append(clust)
    trackerjson['paritition_trials'] = []
    rmse_opt = 10000.

    trials = gen_trials(minclustlen, maxclustlen)

    Natom = data.exyztrj[0].get_number_of_atoms()
    count = 1
    trainlen_last = 0
    print("\n The trials will run in the sampling width interval : ({},{}) \n".format(
        max(trials), min(trials)))

    for Ntrial in tqdm(trials):
        # set(np.random.randint(data.nsample, maxclustlen, size=10)): #range(1,trialsize+2):

        cmd = "rm train.extxyz test.extxyz gap.xml* quip_test.xyz "
        rmout = subprocess.call(cmd, shell=True)
        # print(rmout)
        data.truncate = deepcopy(Ntrial)  # int(np.floor(maxclustlen /Ntrial) )
        print("\n\nBeginning  trial number : {} with a sampling width of {}".format(
            count, data.truncate))
        trainlist, testlist = data.clusterpartition()
        train_lennew = len(trainlist)

        print("\n\nNumber of training and test configs: {} , {}".format(
            len(trainlist), len(testlist)))

        if train_lennew > trainlen_last:
            print("\n {} new learning configs added by the active sampler".format(
                train_lennew - trainlen_last))
            data.writeconfigs()

            bounds = [{'name': 'cutoff',          'type': 'continuous',  'domain': (cutoff[0], cutoff[1])},
                      {'name': 'delta',            'type': 'discrete',
                      'domain': (0.01, 0.1, 1.0)},
                      {'name': 'n_sparse',        'type': 'discrete',
                      'domain': np.arange(sparse[0], sparse[1]+1, 100)},
                      {'name': 'lmax',            'type': 'discrete',
                      'domain': np.arange(lmax[0], lmax[1]+1)},
                      {'name': 'nmax',            'type': 'discrete',  'domain': np.arange(nmax[0], nmax[1]+1)}, ]

            # optimizer
            opt_quip = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=int(Nopt[0]),
                                                           model_type="GP_MCMC",
                                                           acquisition_type='EI_MCMC',  # EI
                                                           evaluator_type="predictive",  # Expected Improvement
                                                           exact_feval=False,
                                                           maximize=False)  # --> True only for r2

            # optimize MP model
            print("\nBegin Optimization run \t")
            opt_quip.run_optimization(max_iter=int(Nopt[1]))
            hyperdict = {}
            for num in range(len(bounds)):
                hyperdict[bounds[num]["name"]] = opt_quip.x_opt[num]
            hyperdict["MAE"] = opt_quip.fx_opt
            trackerjson['paritition_trials'].append(
                {'test': trainlist, 'train': testlist, 'hyperparam': hyperdict})
            # ---> Update only if configs increased over iterations!
            trainlen_last = deepcopy(train_lennew)
            if opt_quip.fx_opt < rmse_opt:
                rmse_opt = float(opt_quip.fx_opt)
                best_train = deepcopy(trainlist)
                best_test = deepcopy(testlist)
                print("\n MAE lowered in this trial: {} eV/Atom".format(rmse_opt/Natom))
                track_metric.append(rmse_opt/Natom)
                best_hyperparam = deepcopy(hyperdict)
                if np.round(rmse_opt/Natom, 4) <= Etol:  # count != 1  and
                    print("\n Optimal configs found! on {}th trial with hyper parameters : {}\n".format(
                        count, best_hyperparam))
                    with open('activelearned_quipconfigs.json', 'w') as outfile:
                        json.dump(trackerjson, outfile, indent=4)
                    print(
                        "\nActive learning history written to 'activelearned_quipconfigs.json' ")

                    train_xyz = []
                    test_xyz = []
                    for i in best_train:
                        train_xyz.append(data.exyztrj[i])
                    for j in best_test:
                        test_xyz.append(data.exyztrj[j])

                    write("opt_train.extxyz", train_xyz)
                    write("opt_test.extxyz", test_xyz)
                    print(
                        "\nActive learnied configurations written to 'opt_train.extxyz','opt_test.extxyz' ")
                    break
        else:
            print("No new configs found in the {} trial. Skipping!".format(count))
        count += 1

    plot_metric(track_metric)