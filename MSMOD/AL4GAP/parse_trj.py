# -*- coding: utf-8 -*-
"""    
@author: Ganesh Sivaraman
"""
def parse_lammps(execpath):
    '''
    Input args:
    execpath : Path to lmp run folder (str)     
    O/P args: 
    extxyz : Extended xyz formatted file
    return:
    A string
    '''
    import numpy as np
    from ase.io import read, write
    import os
    import subprocess

    os.chdir(execpath)
    # execpath = os.getcwd()  ##--> Does not work

    merge_these_files = []  # --> Merge the parsed and cleaned exyz
    xyzlist = []
    for allfiles in os.listdir(execpath):
        if 'salt_npt' in allfiles:
            xyzlist.append(allfiles)
    xyzlist = sorted(xyzlist, reverse=True)

    for count, xyzfile in enumerate(xyzlist):
        if 'salt_npt' in xyzfile:
            print(count, xyzfile, xyzfile[9:][:-4])
            logfile = 'log_' + xyzfile[9:][:-4] + '.npt'
            print("\n" + logfile)
            with open(os.path.join(execpath, logfile), 'rU') as file:
                data = file.readlines()

            for num, line in enumerate(data):
                if 'Step' in line:
                    Start = num
                if 'Loop time' in line:
                    End = num

            extract = data[Start+1:End]
            extract = [elem.strip("\n").split() for elem in extract]
            extract = np.asarray(extract)

            traj = read(os.path.join(execpath, xyzfile), ':')

            print("Trajectory length : {}, Property array length : {}\n".format(
                len(traj), extract.shape[0]))

            for volume, atom in zip(extract[:, 3], traj):
                box = float(volume)**(1/3)
                cell = ((box, 0, 0), (0, box, 0), (0, 0, box))
                atom.set_cell(cell)
                atom.set_pbc([1, 1, 1])

            outfile = os.path.join(
                execpath, 'traj_WithnoE_{}.extxyz'.format(count))
            write(outfile, traj)

            # ---> The final trajectory file to QUIP
            writefile = os.path.join(execpath, "To_QUIP_{}".format(count))

            ind = 0  # ---> To pull the energy value from extract
            if os.path.isfile(outfile):
                with open(outfile, 'rU') as fin:
                    for line in fin:
                        if 'Lattice=' in line:
                            line = line.strip(
                                '\n') + ' energy=' + str(extract[:, 5][ind]) + ' \n'
                            ind += 1

                        else:
                            line = line
                        with open(writefile+'.extxyz', 'a+') as fout:
                            fout.write(line)

            else:
                print(outfile+" not found!")

            merge_these_files.append(writefile+'.extxyz')

            print("\nThe tagged  QUIP extXYZ formatted file is written to : {}.extxyz".format(
                writefile))

    merge = []
    print("\n", merge_these_files)
    for exyz in merge_these_files:
        atomlist = read(exyz, ':')
        [merge.append(atoms) for atoms in atomlist]

    print("\nMerged trajectory length : {} ".format(len(merge)))

    parse_limit = 33001
    if 'deform' in execpath:
        parse_limit = 25001

    i = 0
    while len(merge) > parse_limit :
        merge = merge[::2]
        i += 1
        print('\n', i, len(merge))

    print("\n Number of reduced  confs: {}".format(len(merge)))
    write("To_QUIP.extxyz", merge)

    print("\n Removing intermediate parsed files!")
    cmd = "rm  traj_WithnoE_* To_QUIP_*  "
    rmout = subprocess.call(cmd, shell=True)

    return 'Done'

parse_lammps(".")

def parse_merge_plot_AL(pathlist, writepath):
    '''
    Input args:
    pathlist : Path to AL  folder (list)     
    writepath : Path to write merged files (str)
    O/P args: 
    extxyz : Extended xyz formatted file
    return:
    A string
    '''

    from ase.io import read, write
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('brg')
    plt.rcParams["font.family"] = "Arial"
    plt.style.use('seaborn')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    test, train = list(), list()
    for folder in pathlist:
        os.chdir(folder)
        ltrain = read(folder + '/opt_train.extxyz', ':')
        ltest = read(folder + '/opt_test.extxyz', ':')
        [train.append(atoms) for atoms in ltrain]
        [test.append(atoms) for atoms in ltest]

    write(os.path.join(writepath, 'train.extxyz'), train)
    write(os.path.join(writepath, 'test.extxyz'), test)

    density_train, density_test = list(), list()
    [density_train.append(sum(conf.get_masses()) /
                          (0.6023*conf.get_volume())) for conf in train]
    [density_test.append(sum(conf.get_masses()) /
                         (0.6023*conf.get_volume())) for conf in test]

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    n, bins, patches = ax1.hist(
        density_train, 25, facecolor='green', alpha=0.5, ls='dashed')
    n, bins, patches = ax2.hist(
        density_test, 25, facecolor='red', alpha=0.5, ls='dotted')
    ax1.set_xlim([0.8, 2.25])
    ax1.set_ylim([1, 100])
    ax2.set_xlim([0.8, 2.25])
    ax2.set_ylim([1, 100])

    ax1.set_ylabel('Count', fontsize=20)
    ax2.set_ylabel('Count', fontsize=20)
    plt.xlabel('Density g. cm-3', fontsize=20)
    plt.grid('True')
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(writepath, 'density.png'), dpi=300)

    return 'Done'


def parse_vasp(pathlist, writepath):
    '''
    Input args:
    pathlist : Path to Vasp O/P  folder (list)     
    writepath : Path to write merged files (str)
    O/P args: 
    extxyz : Extended xyz formatted file
    return:
    A string
    '''

    from ase.io import read, write
    import os

    vaspextxyz = list()
    for folder in pathlist:
        os.chdir(folder)
        xmlpath = os.path.join(folder, 'vasprun.xml')
        lxyz = read(xmlpath, ':')
        if lxyz[0].get_potential_energy() <= 0.:
            vaspextxyz.append(lxyz[0])
        elif lxyz[0].get_potential_energy() > 0:
            print("\n Unbounded energy encountered for config no: {} with E : {} ".format(
                count+1, lxyz[0].get_potential_energy()))

    write(os.path.join(writepath, 'vasp_output.extxyz'), vaspextxyz)

    return 'Done'
