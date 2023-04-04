# -*- coding: utf-8 -*-
"""    
@authors: Nicholas Jackson, Ganesh Sivaraman
"""
import os
import numpy as np


class TosiFumiParams:

    def __init__(self, composition_list, mol_fraction_list):
        self.composition_list = composition_list
        self.mol_fraction_list = mol_fraction_list
        # 10^-60 erg cm^6 = 0.624150648 eV Ang^6 is the correct conversion for C
        #
        # [c++,c--,c+-]
        self.cations = ['Li', 'Na', 'K', 'Rb', 'Cs']
        self.anions = ['F', 'Cl', 'Br', 'I']
        self.allatoms = self.cations + self.anions
        self.disp_scale = 0.624150648
        self.C_dict = {}
        self.C_dict['Li,F'] = {'Li,Li': 0.073, 'FF,F': 14.5, 'Li,F': 0.8}
        self.C_dict['Li,Cl'] = {'Li,Li': 0.073, 'Cl,Cl': 111.0, 'Li,Cl': 2.0}
        self.C_dict['Li,Br'] = {'Li,Li': 0.073, 'Br,Br': 185.0, 'Li,Br': 2.5}
        self.C_dict['Li,I'] = {'Li,Li': 0.073, 'I,I': 378.0, 'Li,I': 3.3}
        self.C_dict['Na,F'] = {'Na,Na': 1.68, 'F,F': 16.5, 'Na,F': 4.5}
        self.C_dict['Na,Cl'] = {'Na,Na': 1.68, 'Cl,Cl': 116.0, 'Na,Cl': 11.2}
        self.C_dict['Na,Br'] = {'Na,Na': 1.68, 'Br,Br': 196.0, 'Na,Br': 14.0}
        self.C_dict['Na,I'] = {'Na,Na': 1.68, 'I,I': 392.0, 'Na,I': 19.1}
        self.C_dict['K,F'] = {'K,K': 24.3, 'F,F': 18.6, 'K,F': 19.5}
        self.C_dict['K,Cl'] = {'K,K': 24.3, 'Cl,Cl': 124.5, 'K,Cl': 48.0}
        self.C_dict['K,Br'] = {'K,K': 24.3, 'Br,Br': 206.0, 'K,Br': 60.0}
        self.C_dict['K,I'] = {'K,K': 24.3, 'I,I': 403.0, 'K,I': 82.0}
        self.C_dict['Rb,F'] = {'Rb,Rb': 59.4, 'F,F': 18.9, 'Rb,F': 31.0}
        self.C_dict['Rb,Cl'] = {'Rb,Rb': 59.4, 'Cl,Cl': 130.0, 'Rb,Cl': 79.0}
        self.C_dict['Rb,Br'] = {'Rb,Rb': 59.4, 'Br,Br': 215.0, 'Rb,Br': 99.0}
        self.C_dict['Rb,I'] = {'Rb,Rb': 59.4, 'I,I': 428.0, 'Rb,I': 135.0}
        self.C_dict['Cs,F'] = {'Cs,Cs': 152.0, 'F,F': 19.1, 'Cs,F': 52.0}
        self.C_final_dict = {}
        # D units are 10^-76 ergs.cm^8
        # So the correct conversion is also 0.624150648
        # [d++,d--,d+-]
        self.D_dict = {}
        self.D_dict['Li,F'] = {'Li,Li': 0.03, 'F,F': 17.0, 'Li,F': 0.6}
        self.D_dict['Li,Cl'] = {'Li,Li': 0.03, 'Cl,Cl': 223.0, 'Li,Cl': 2.4}
        self.D_dict['Li,Br'] = {'Li,Li': 0.03, 'Br,Br': 423.0, 'Li,Br': 3.3}
        self.D_dict['Li,I'] = {'Li,Li': 0.03, 'I,I': 1060.0, 'Li,I': 5.3}
        self.D_dict['Na,F'] = {'Na,Na': 0.8, 'F,F': 20.0, 'Na,F': 3.8}
        self.D_dict['Na,Cl'] = {'Na,Na': 0.8, 'Cl,Cl': 233.0, 'Na,Cl': 13.9}
        self.D_dict['Na,Br'] = {'Na,Na': 0.8, 'Br,Br': 450.0, 'Na,Br': 19.0}
        self.D_dict['Na,I'] = {'Na,Na': 0.8, 'I,I': 1100.0, 'Na,I': 31.0}
        self.D_dict['K,F'] = {'K,K': 24.0, 'F,F': 22.0, 'K,F': 21.0}
        self.D_dict['K,Cl'] = {'K,K': 24.0, 'Cl,Cl': 250.0, 'K,Cl': 73.0}
        self.D_dict['K,Br'] = {'K,K': 24.0, 'Br,Br': 470.0, 'K,Br': 99.0}
        self.D_dict['K,I'] = {'K,K': 24.0, 'I,I': 1130.0, 'K,I': 156.0}
        self.D_dict['Rb,F'] = {'Rb,Rb': 82.0, 'F,F': 23.0, 'Rb,F': 40.0}
        self.D_dict['Rb,Cl'] = {'Rb,Rb': 82.0, 'Cl,Cl': 260.0, 'Rb,Cl': 134.0}
        self.D_dict['Rb,Br'] = {'Rb,Rb': 82.0, 'Br,Br': 490.0, 'Rb,Br': 180.0}
        self.D_dict['Rb,I'] = {'Rb,Rb': 82.0, 'I,I': 1200.0, 'Rb,I': 280.0}
        self.D_dict['Cs,F'] = {'Cs,Cs': 278.0, 'F,F': 23.0, 'Cs,F': 78.0}
        self.D_final_dict = {}
        # Ion sizes taken from 1963 Tosi Fumi paper in J. Phys. Chem. Solids
        #Units in Angstroms
        self.sig_dict = {}
        self.sig_dict['Li'] = 0.816
        self.sig_dict['Na'] = 1.170
        self.sig_dict['K'] = 1.463
        self.sig_dict['Rb'] = 1.587
        self.sig_dict['Cs'] = 1.720
        self.sig_dict['F'] = 1.179
        self.sig_dict['Cl'] = 1.585
        self.sig_dict['Br'] = 1.716
        self.sig_dict['I'] = 1.907
        self.sig_final_dict = {}
        # Compute all sigmas using ionic radii
        for i in range(len(self.allatoms)):
            typei = self.allatoms[i]
            for j in range(len(self.allatoms)):
                typej = self.allatoms[j]
                self.sig_final_dict[(
                    typei, typej)] = self.sig_dict[typei] + self.sig_dict[typej]
        # Rho parameters taken from Tosi-Fumi as well
        #Units in Angstroms
        self.rho_dict = {}
        self.rho_dict['Li,F'] = 0.299
        self.rho_dict['Li,Cl'] = 0.342
        self.rho_dict['Li,Br'] = 0.353
        self.rho_dict['Li,I'] = 0.430
        self.rho_dict['Na,F'] = 0.330
        self.rho_dict['Na,Cl'] = 0.317
        self.rho_dict['Na,Br'] = 0.340
        self.rho_dict['Na,I'] = 0.386
        self.rho_dict['K,F'] = 0.338
        self.rho_dict['K,Cl'] = 0.337
        self.rho_dict['K,Br'] = 0.335
        self.rho_dict['K,I'] = 0.355
        self.rho_dict['Rb,F'] = 0.328
        self.rho_dict['Rb,Cl'] = 0.318
        self.rho_dict['Rb,Br'] = 0.335
        self.rho_dict['Rb,I'] = 0.337
        self.rho_dict['Cs,F'] = 0.282
        # Mixing formula for the system rho
        rho_1 = self.rho_dict[self.composition_list[0]]
        rho_2 = self.rho_dict[self.composition_list[1]]
        self.system_rho = 1 / \
            ((self.mol_fraction_list[0]/rho_1) +
             (self.mol_fraction_list[1]/rho_2))
        # dict for Pauling factor contributions Zi/ni
        # Pauling factor contribution term B_ij = b*(1 + Zi/ni + Zj/ni)
        self.Zi_over_ni_dict = {}
        self.Zi_over_ni_dict['F'] = -1/8.
        self.Zi_over_ni_dict['Cl'] = -1/8.
        self.Zi_over_ni_dict['Br'] = -1/8.
        self.Zi_over_ni_dict['I'] = -1/8.
        self.Zi_over_ni_dict['Li'] = 1/2.
        self.Zi_over_ni_dict['Na'] = 1/8.
        self.Zi_over_ni_dict['K'] = 1/8.
        self.Zi_over_ni_dict['Rb'] = 1/8.
        self.Zi_over_ni_dict['Cs'] = 1/8.
        self.Aij_final_dict = {}
        self.bconstant = 0.2109796
        for i in range(len(self.allatoms)):
            typei = self.allatoms[i]
            for j in range(len(self.allatoms)):
                typej = self.allatoms[j]
                self.Aij_final_dict[(typei, typej)] = self.bconstant * \
                    (1.+self.Zi_over_ni_dict[typei] +
                     self.Zi_over_ni_dict[typej])

    def mix_dispersion_parameters(self):
        # This is only configured to work for binary mixtures at the moment
        # This assumes the mixing rules from Journal of Molecular Liquids 209 (2015) 498–507
        if len(self.composition_list) == 2:
            self.unique_atom_types = []
            self.redundant_atom_types = []
            # Find all unique atom types in the system
            # Also find any redundant atoms for which mixing rules will need to be applied
            for el in self.composition_list:
                atoms = el.split(',')
                for a in atoms:
                    if a not in self.unique_atom_types:
                        self.unique_atom_types.append(a)
                    else:
                        self.redundant_atom_types.append(a)
            # Set up dictionary for C and D
            for i in range(len(self.unique_atom_types)):
                typei = self.unique_atom_types[i]
                for j in range(len(self.unique_atom_types)):
                    typej = self.unique_atom_types[j]
                    self.C_final_dict[(typei, typej)] = 0.0
                    self.D_final_dict[(typei, typej)] = 0.0
            # Now loop through these dictionaries
            for el1, el2 in self.C_final_dict.keys():
                # Mix anions
                if el1 in self.anions and el2 in self.anions and el1 == el2:
                    Cparam1 = self.C_dict[self.composition_list[0]
                                          ][el1+','+el1]
                    Cparam2 = self.C_dict[self.composition_list[1]
                                          ][el1+','+el1]
                    Dparam1 = self.D_dict[self.composition_list[0]
                                          ][el1+','+el1]
                    Dparam2 = self.D_dict[self.composition_list[1]
                                          ][el1+','+el1]
                    newC = self.mol_fraction_list[0]*Cparam1 + \
                        self.mol_fraction_list[1]*Cparam2
                    newD = self.mol_fraction_list[0]*Dparam1 + \
                        self.mol_fraction_list[1]*Dparam2
                    self.C_final_dict[(el1, el2)] = self.disp_scale*newC
                    self.D_final_dict[(el1, el2)] = self.disp_scale*newD
                else:
                    if el1 in self.composition_list[0] and el2 in self.composition_list[0]:
                        tempCdict = self.C_dict[self.composition_list[0]]
                        tempDdict = self.D_dict[self.composition_list[0]]
                        if el1+','+el2 not in tempCdict.keys():
                            Cparam = tempCdict[el2+','+el1]
                            Dparam = tempDdict[el2+','+el1]
                        else:
                            Cparam = tempCdict[el1+','+el2]
                            Dparam = tempDdict[el1+','+el2]
                        self.C_final_dict[(el1, el2)] = self.disp_scale*Cparam
                        self.D_final_dict[(el1, el2)] = self.disp_scale*Dparam
                    elif el1 in self.composition_list[1] and el2 in self.composition_list[1]:
                        tempCdict = self.C_dict[self.composition_list[1]]
                        tempDdict = self.D_dict[self.composition_list[1]]
                        if el1+','+el2 not in tempCdict.keys():
                            Cparam = tempCdict[el2+','+el1]
                            Dparam = tempDdict[el2+','+el1]
                        else:
                            Cparam = tempCdict[el1+','+el2]
                            Dparam = tempDdict[el1+','+el2]
                        self.C_final_dict[(el1, el2)] = self.disp_scale*Cparam
                        self.D_final_dict[(el1, el2)] = self.disp_scale*Dparam
                    elif el1 in self.composition_list[0] and el2 in self.composition_list[1]:
                        C1 = self.disp_scale * \
                            self.C_dict[self.composition_list[0]][el1+','+el1]
                        C2 = self.disp_scale * \
                            self.C_dict[self.composition_list[1]][el2+','+el2]
                        D1 = self.disp_scale * \
                            self.D_dict[self.composition_list[0]][el1+','+el1]
                        D2 = self.disp_scale * \
                            self.D_dict[self.composition_list[1]][el2+','+el2]
                        self.C_final_dict[(el1, el2)] = (C1*C2)**0.5
                        self.D_final_dict[(el1, el2)] = (D1*D2)**0.5
                    elif el1 in self.composition_list[1] and el2 in self.composition_list[0]:
                        C1 = self.disp_scale * \
                            self.C_dict[self.composition_list[1]][el1+','+el1]
                        C2 = self.disp_scale * \
                            self.C_dict[self.composition_list[0]][el2+','+el2]
                        D1 = self.disp_scale * \
                            self.D_dict[self.composition_list[1]][el1+','+el1]
                        D2 = self.disp_scale * \
                            self.D_dict[self.composition_list[0]][el2+','+el2]
                        self.C_final_dict[(el1, el2)] = (C1*C2)**0.5
                        self.D_final_dict[(el1, el2)] = (D1*D2)**0.5

    def return_parameter_set(self):
        print('Parameters for {} and {}'.format(
            self.composition_list[0], self.composition_list[1]))
        print('for mol fractions of {} and {}'.format(
            self.mol_fraction_list[0], self.mol_fraction_list[1]))
        print('type1 type2 Aij sigma rho C D')
        final_param_dict = {}
        for el1, el2 in self.C_final_dict.keys():
            Aij = round(self.Aij_final_dict[(el1, el2)], 6)
            sig = round(self.sig_final_dict[(el1, el2)], 6)
            rho = round(self.system_rho, 6)
            C = round(self.C_final_dict[(el1, el2)], 6)
            D = round(self.D_final_dict[(el1, el2)], 6)
            final_param_dict[(el1, el2)] = [Aij, sig, rho, C, D]
            print('{} {} {} {} {} {} {} '.format(
                el1, el2, Aij, sig, rho, C, D))
        return final_param_dict
