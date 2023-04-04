# -*- coding: utf-8 -*-
"""    
@authors: Nicholas Jackson, Ganesh Sivaraman
"""
import os
import numpy as np
import random
from .tosi_fumi_params import *


class simbox():
    def __init__(self):
        # OPLSAA vdw and charge parameters
        # First param sigma, second epsilon
        # Units angstrom and kcal/mol
        # Values obtained from TINKER's oplsaa.prm file
        # B_dict corresponds to the A parameter in lammps born/coul/long pair_style definition
        # Bij = (1 + Z_i/n_i + Z_j/n_j)*0.338*10^-19 J
        # Zi = atomic number, ni = # valence electrons
        # sigma_ij = sigma_i + sigma_j #sigma = crystal ionic radii
        # rho = hardness parameter
        #C = dipole-dipole
        #D = dipole-quadrupole
        # Dispersion parameters from  J. Chem. Phys. 1, 270 (1933); https://doi.org/10.1063/1.1749283
        # C units 10^-60 ergs cm^6
        # D units 10^-76 ergs cm^6
        self.kcaltoeV = 0.043
        self.chg_dict = {}
        self.chg_dict['F'] = -1.0
        self.chg_dict['Cl'] = -1.0
        self.chg_dict['Br'] = -1.0
        self.chg_dict['I'] = -1.0
        self.chg_dict['Li'] = 1.0
        self.chg_dict['Na'] = 1.0
        self.chg_dict['K'] = 1.0
        self.chg_dict['Rb'] = 1.0
        self.chg_dict['Cs'] = 1.0
        self.mass_dict = {}
        self.mass_dict['F'] = 18.998
        self.mass_dict['Cl'] = 35.453
        self.mass_dict['Br'] = 79.904
        self.mass_dict['I'] = 126.905
        self.mass_dict['Li'] = 6.941
        self.mass_dict['Na'] = 22.990
        self.mass_dict['K'] = 39.098
        self.mass_dict['Rb'] = 85.468
        self.mass_dict['Cs'] = 132.905

    def set_composition(self, composition, fractions):
        # First argument is a list of elements, entered as strings, e.g.
        # ['Na','Br','Sr','F']
        # Second argument is the fractions of each
        # [0.1,0.5,0.3,0.1]
        if len(composition) != len(fractions):
            print('Composition and fraction lengths do not match. Exiting...')
            exit()
        if np.round(sum(fractions), 2) != 1.:
            print('Elemental fractions do not sum to 1. Exiting...')
            exit()
        self.elemental_comp = composition
        self.elemental_frac = fractions
        TFGen = TosiFumiParams(self.elemental_comp, self.elemental_frac)
        TFGen.mix_dispersion_parameters()
        self.TFparams = TFGen.return_parameter_set()
        # self.atom_comp is the renamed elemental_comp from the old version
        # self.atom_frac is the renamed elemental_frac from the old version
        self.atom_comp = []
        self.atom_frac = []
        # Get all unique atoms in elemental_comp
        for i in range(len(self.elemental_comp)):
            salt = self.elemental_comp[i]
            for a in salt.split(','):
                if a not in self.atom_comp:
                    self.atom_comp.append(a)
        self.num_atom_types = len(self.atom_comp)
        self.frac_dict = {}
        # Now compute the atom-wise elemental fractions
        # Initialize a dictionary to store atom-wise fractions
        for el in self.atom_comp:
            self.frac_dict[el] = 0.0
        # convert elemental_frac to atom-wise terms
        for i in range(len(self.elemental_comp)):
            salt = self.elemental_comp[i]
            for a in salt.split(','):
                self.frac_dict[a] += self.elemental_frac[i]/2.
        for el in self.atom_comp:
            self.atom_frac.append(self.frac_dict[el])
        print('\n')
        print('Elemental composition is:')
        print(self.atom_comp)
        print(self.atom_frac)

    def set_target_number_of_atoms(self, N):
        self.tot_atom_target = N
        min_frac = np.min(self.atom_frac)
        self.sim_atom_count = np.array(
            [int(i/min_frac) for i in self.atom_frac])
        self.tot_atoms = np.sum(self.sim_atom_count)
        multiple = self.tot_atom_target//self.tot_atoms
        self.sim_atom_count *= multiple  # Number of atoms of each type
        # Total number of atoms in system
        self.tot_atoms = np.sum(self.sim_atom_count)

    # Need to run set_target_number_atoms before set_density
    # Takes density in units of kg/m^3
    def set_density(self, dens, epsilon=0.3, dv=250, deform=False):
        self.dens = dens
        self.epsilon = epsilon  # Epsilon paramter for %  deform the box
        self.dv = dv  # change in volume
        self.deform = deform
        self.total_mass = 0.0  # Compute total mass for target atom number
        for atom, num in zip(self.atom_comp, self.sim_atom_count):
            mass = self.mass_dict[atom]
            self.total_mass += mass*num
        self.total_mass *= 1.66054*10**-27  # Convert amu to kg
        # Compute box length in meters
        self.boxl = (self.total_mass / self.dens)**(0.33333)
        self.boxl *= 10**10  # Convert meters to angstroms
        self.boxl = np.round(self.boxl, 6)
        self.xlo = 0.0  # Set MD box dimensions
        self.ylo = 0.0
        self.zlo = 0.0
        if self.deform == False:
            self.xhi = self.boxl
            self.yhi = self.boxl
            self.zhi = self.boxl
            self.vol1 = np.round(self.boxl ** 3)

        elif self.deform == True:
            upper = self.dens + self.dens * epsilon
            lower = self.dens - self.dens * epsilon
            # Compute deformation target box length in meters
            self.box2 = (self.total_mass / lower)**(0.33333)
            self.box2 *= 10**10  # Convert meters to angstroms
            self.box2 = np.round(self.box2, 6)
            self.vol2 = np.round(self.box2 ** 3)

            # Compute deformation target box length in meters
            self.box3 = (self.total_mass / upper)**(0.33333)
            self.box3 *= 10**10  # Convert meters to angstroms
            self.box3 = np.round(self.box3, 6)
            self.xhi = self.box3
            self.yhi = self.box3
            self.zhi = self.box3
            self.vol3 = np.round(self.box3 ** 3)

    def make_initial_config(self, min_dist=0.5, writeconfig=1):
        # x,y,z type
        # Can use packmol for this later, but doing this manually
        # Because its trivial for elemental systems
        print('Making initial configuration...')
        self.type = np.array([])
        self.xyz = np.array([[0, 0, 0]])
        for i in range(self.num_atom_types):  # Loop over atom types
            atom = self.atom_comp[i]
            num = self.sim_atom_count[i]
            for j in range(num):  # Loop over number of atoms of each type
                truth = 0
                packing_count = 0
                while truth == 0:
                    xtrial = random.uniform(self.xlo, self.xhi)
                    ytrial = random.uniform(self.ylo, self.yhi)
                    ztrial = random.uniform(self.zlo, self.zhi)
                    self.xyz = np.append(self.xyz, np.array(
                        [[xtrial, ytrial, ztrial]]), axis=0)
                    rij = self.xyz[:, np.newaxis, :] - \
                        self.xyz[np.newaxis, :, :]
                    rij = rij - self.boxl*np.rint(rij/self.boxl)
                    rij_mag = np.sqrt(np.sum(rij**2., axis=-1))
                    rij_mag = rij_mag[np.triu_indices_from(rij_mag, k=1)]
                    if np.min(rij_mag) >= min_dist:
                        truth = 1
                        if i == 0 and j == 0:  # Delete the first element I used to jumpstart packing
                            self.xyz = np.delete(self.xyz, 0, 0)
                    else:
                        self.xyz = np.delete(self.xyz, -1, 0)
                        packing_count += 1
                    if packing_count == 1000:
                        print(
                            'Could not pack all atoms with the required minimum distance.\n')
                        print('Try decreasing min_dist. Exiting...\n')
                        exit()
                self.type = np.append(self.type, np.array([i]))
        print('Packing successful!')
        if writeconfig == 1:
            with open('init_config.xyz', 'w') as f:
                f.write('{} \n'.format(self.tot_atoms))
                f.write('{} {} initial configuration file \n'.format(
                    self.atom_comp, self.atom_frac))
                for i in range(self.tot_atoms):
                    atomtype = int(self.type[i])
                    element = self.atom_comp[atomtype]
                    x, y, z = self.xyz[i]
                    f.write('{} \t {:6f} \t {:6f} \t {:6f} \n'.format(
                        element, x, y, z))

    def write_lammps_datafile(self, filename):
        print('Writing LAMMPS data file...')
        with open(filename, 'w') as f:
            f.write('LAMMPS data file\n')
            f.write('\n')
            f.write('{} atoms\n'.format(self.tot_atoms))
            f.write('{} atom types\n'.format(self.num_atom_types))
            f.write('0 bonds\n')
            f.write('0 bond types\n')
            f.write('0 angles\n')
            f.write('0 angle types\n')
            f.write('0 dihedrals\n')
            f.write('0 dihedral types\n')
            f.write('0 impropers\n')
            f.write('0 improper types\n')
            f.write('\n')
            f.write('{:6f} \t {:6f} \t xlo xhi\n'.format(self.xlo, self.xhi))
            f.write('{:6f} \t {:6f} \t ylo yhi\n'.format(self.ylo, self.yhi))
            f.write('{:6f} \t {:6f} \t zlo zhi\n'.format(self.zlo, self.zhi))
            f.write('\n')
            f.write('Masses\n')
            f.write('\n')
            for i in range(self.num_atom_types):
                element = self.atom_comp[i]
                mass = self.mass_dict[element]
                f.write('{} {}\n'.format(i+1, mass))
            f.write('\n')
            f.write('Atoms\n')
            f.write('\n')
            for i in range(self.tot_atoms):
                atmnum = i+1
                mol = i+1
                type = int(self.type[i]+1)
                element = self.atom_comp[int(self.type[i])]
                chg = self.chg_dict[element]
                x, y, z = self.xyz[i]
                f.write('{} \t {} \t {} \t {} \t {} \t {} \n'.format(
                    atmnum, type, chg, x, y, z))
            f.write('\n')

    def set_lammps_params(self, print_reg):
        self.print_reg = print_reg
        self.unit_style = 'metal'
        self.atom_style = 'charge'
        self.kspace_style = 'pppm'
        self.kspace_acc = 1.0e-4  # -->1.0e-5
        if self.deform == False:
            self.noncoul_cut = np.floor(self.boxl / 2.)
            self.coul_cut = np.floor(self.boxl / 2.)
        elif self.deform == True:
            self.noncoul_cut = np.floor(min(self.box2, self.box3) / 2.)
            self.coul_cut = np.floor(min(self.box2, self.box3) / 2.)

    def write_lammps_inputfile(self, EquilTempStart, EquilTempStop, SampleTemp, tstep, melt_steps, equil_steps, sample_steps, nsims, datafile, filename):
        self.tstep = tstep
        self.EquilTempStart = EquilTempStart
        self.EquilTempStop = EquilTempStop
        self.SampleTemp = SampleTemp
        self.melt_steps = melt_steps
        self.equil_steps = equil_steps
        self.sample_steps = sample_steps
        self.nsims = nsims
        print('Writing LAMMPS input file...')
        with open(filename, 'w', encoding="utf-8") as f:
            f.write('units {}\n'.format(self.unit_style))
            f.write('atom_style {}\n'.format(self.atom_style))
            f.write(
                'pair_style \t born/coul/long {} {} \n'.format(self.noncoul_cut, self.coul_cut))
            f.write('boundary p p p \n')
            f.write('kspace_style {} {} \n'.format(
                self.kspace_style, self.kspace_acc))
            f.write('pair_modify shift yes \n')
            f.write('read_data {}\n'.format(datafile))
            f.write('\n')
            f.write('variable my_temp equal temp\n')
            f.write('variable my_rho equal density\n')
            f.write('variable my_vol equal vol\n')
            f.write('\n')
            f.write('#Pair Coeffs\n')
            f.write('\n')
            for i in range(self.num_atom_types):
                element_i = self.atom_comp[i]
                for j in range(i, self.num_atom_types):
                    element_j = self.atom_comp[j]
                    B, sig, rho, C, D = self.TFparams[(element_i, element_j)]
                    f.write('pair_coeff {} {} {} {} {} {} {}\n'.format(
                        i+1, j+1, B, rho, sig, C, D))
            f.write('\n')
            f.write('#MINIMIZATION\n')
            f.write('minimize 1.0e-24 1.0e-24 10000 10000\n')
            f.write('timestep {} \n'.format(self.tstep))
            f.write('neighbor 3.0 bin\n')
            f.write(
                'neigh_modify delay 10 every 10 check yes page 100000 one 10000\n')
            f.write('\n')
            f.write('velocity all create {} {}\n'.format(
                self.EquilTempStart, random.randint(1, 1000000)))
            f.write('\n')

            f.write('#Melt-Quench RUN\n')
            f.write('fix mynvt all nvt temp {} {} 1\n'.format(
                self.EquilTempStart, self.EquilTempStart))
            f.write('thermo {}\n'.format(self.print_reg))
            f.write(
                'thermo_style custom step temp density vol press etotal ke pe evdwl ecoul elong\n')
            f.write('log log.melt\n')
            f.write('run {}\n'.format(self.melt_steps))
            f.write('unfix mynvt\n')
            f.write('fix mynvt all nvt temp {} {} 1\n'.format(
                self.EquilTempStart, self.EquilTempStop))
            f.write('thermo {}\n'.format(self.print_reg))
            f.write(
                'thermo_style custom step temp density vol press etotal ke pe evdwl ecoul elong\n')
            f.write('log log.quench\n')
            f.write('run {}\n'.format(self.melt_steps))
            f.write('unfix mynvt\n')
            f.write('\n')

            f.write('#EQUILIBRATION RUN\n')
            #f.write('fix mymomentum all momentum 1000 linear 1 1 1 angular\n')
            if self.deform == True:
                f.write('fix mynvt all nvt temp {} {} 1\n'.format(
                    self.EquilTempStop, self.EquilTempStop))
            elif self.deform == False:
                #f.write('fix mynvt all npt temp {} {} 0.5 iso 0.0 0.0 10.0\n'.format(self.EquilTempStop,self.EquilTempStop))
                f.write('fix mynvt all nvt temp {} {} 0.5 \n'.format(
                    self.EquilTempStop, self.EquilTempStop))
            f.write('thermo {}\n'.format(self.print_reg))
            f.write(
                'thermo_style custom step temp density vol press etotal ke pe evdwl ecoul elong\n')
            f.write('log log.equi\n')
            f.write('run {}\n'.format(self.equil_steps))
            f.write('unfix mynvt\n')
            #f.write('unfix mymomentum\n')
            f.write('\n')
            f.write('#NPT SAMPLING AND PRINTING TRAJECTORY\n')
            #f.write('fix mymomentum all momentum 1000 linear 1 1 1 angular\n')
            f.write('reset_timestep 0\n')
            if self.deform == True:
                # ------ Added the loops below-------------
                f.write('\n')
                f.write('variable vollo equal {}\n'.format(self.vol3))
                f.write('variable volhi equal {}\n'.format(self.vol2))
                f.write('variable llo equal v_vollo^0.333333333\n')
                f.write('variable lhi equal v_volhi^0.333333333\n')
                f.write('variable ttotal equal {}\n'.format(self.sample_steps))
                f.write('variable nsims equal {}\n'.format(self.nsims))
                f.write('variable timepersim equal v_ttotal/v_nsims\n')
                f.write('fix mynvt all nvt temp {} {} 1 #Keep the temp fix constant and outside of the loop\n'.format(
                    self.SampleTemp, self.SampleTemp))
                f.write('label loopa\n')
                f.write('variable a loop ${nsims}\n')

                # OLD
#                f.write('variable boxl equal (v_lhi-v_llo)*(((v_a*v_timepersim)/v_ttotal)^(0.333333))+v_llo\n')
                # NEW
                f.write(
                    'variable boxl equal (v_vollo+(v_volhi-v_vollo)*(v_a/v_nsims))^(0.333333)\n')
                f.write('print " "\n')
                f.write('print "Loop $a"\n')
                f.write('print "Current Box Length: ${boxl}"\n')
                f.write('print " "\n')
                f.write(
                    'fix mydeform all deform 1 x final 0.0 ${boxl} y final 0.0 ${boxl} z final 0.0 ${boxl}\n')
                f.write('thermo {}\n'.format(self.print_reg))
                f.write(
                    'thermo_style custom step temp density vol press etotal ke pe evdwl ecoul elong\n')
                f.write(
                    'fix averages all ave/time 10 50 1000 v_my_temp v_my_rho v_my_vol file thermo_${boxl}.avg\n')
                f.write('dump myxyz all xyz {}'.format(
                    self.print_reg) + '  salt_npt_${boxl}.xyz\n')
                f.write('dump_modify myxyz element')
                for el in self.atom_comp:
                    f.write(' {}'.format(el))
                f.write('\n')
                f.write('dump_modify myxyz sort id\n')
                f.write('log log_${boxl}.npt\n')
                f.write('run ${timepersim}\n')
                f.write('unfix mydeform \n')
                f.write('undump myxyz \n')
                f.write('next a\n')
                f.write('jump SELF loopa \n')

            elif self.deform == False:
                f.write('\n')
                f.write('fix mynvt all nvt temp {} {} 0.5 #Keep the temp fix constant and outside of the loop\n'.format(
                    self.SampleTemp, self.SampleTemp))
                f.write('thermo {}\n'.format(self.print_reg))
                f.write(
                    'thermo_style custom step temp density vol press etotal ke pe evdwl ecoul elong\n')
                f.write(
                    'fix averages all ave/time 10 50 1000 v_my_temp v_my_rho v_my_vol file thermo_14.3191176070878.avg\n')
                f.write('dump myxyz all xyz {}'.format(
                    self.print_reg) + '  salt_npt_14.3191176070878.xyz\n')
                f.write('dump_modify myxyz element')
                for el in self.atom_comp:
                    f.write(' {}'.format(el))
                f.write('\n')
                f.write('dump_modify myxyz sort id\n')
                f.write('log log_14.3191176070878.npt\n')
                f.write('run {}\n'.format(self.sample_steps))
                f.write('undump myxyz \n')

            f.write('unfix mynvt \n')
            f.write('write_data final_config.data\n')
