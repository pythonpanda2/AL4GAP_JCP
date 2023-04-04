# -*- coding: utf-8 -*-
"""    
@author:  Ganesh Sivaraman
"""

def setup_inputs(inputdf, ffparam='TF'):
    '''
    Input args:
    inputdf: Experiment phase space in pandas df format
    ffparam: TF (default), OPLS 
    return:
    runpath, lmpFilelist : path to runs folder (list), lammps run file (list) 
    '''

    import os

    cwd = os.getcwd()
    #csv = pd.read_csv(os.path.join(cwd,inputcsv))
    df_clean = inputdf.drop_duplicates(
        subset=list(inputdf.columns)[:-2], keep='first')

    if not os.path.isdir(os.path.join(cwd, 'Data')):
        os.mkdir(os.path.join(os.path.join(cwd, 'Data')))

    DataPath = os.path.join(os.path.join(cwd, 'Data'))

    lmpFilelist = []
    runpath = []

    count = 0
    for ind, row in df_clean.iterrows():
        count += 1

        if not os.path.isdir(os.path.join(DataPath, str(count))):
            os.mkdir(os.path.join(DataPath, str(count)))

        WritePath = os.path.join(DataPath, str(count))
        runpath.append(WritePath)

        # Quip species support
        species = '{'
        if ffparam == 'OPLS':
            composition = list(row.keys())[:-2]
            fraction = list(row.values[:-2])
            density = row.values[-1]

            if 0.0 in fraction:
                zerofrac = fraction.index(0.0)  # --> Remove Zero fraction
                # --> Remove item with zero composition fraction!
                del composition[zerofrac]
                del fraction[zerofrac]

            for val in composition:
                species += val + ':0.0:'

            species = species[:-1]
            species = species + '}'

        elif ffparam == 'TF':
            # Becuase the input to TF class is different
            composition = list(row.keys())[:-3]
            fraction = list(row.values[:-3])
            density = row.values[-1]
            # To multiply the composition list to replicate list
            width = len(fraction)

            if 0.0 in fraction:
                zerofrac = fraction.index(0.0)  # --> Remove Zero fraction
                del composition[zerofrac]
                del fraction[zerofrac]

                for val in composition:
                    species += val + ':0.0:'
                species += 'Cl:0.0}'
                composition = [elem + ',Cl' for elem in composition]
                composition *= width
                fraction *= width

            else:
                for val in composition:
                    species += val + ':0.0:'
                species += 'Cl:0.0}'
                composition = [elem + ',Cl' for elem in composition]
                fraction = [2*frac for frac in fraction]

        filename = 'species.dat'
        print(composition, fraction)
        if not os.path.isfile(os.path.join(WritePath, filename)):
            with open(os.path.join(WritePath, filename), 'w') as f:
                f.write(species)
        print('\nSpecies support for QUIP Written to : {}\n'.format(
            os.path.join(WritePath, filename)))

        if ffparam == 'OPLS':
            from .moltensalt_gen import simbox
            box = simbox()
            box.set_composition(composition, fraction)
            box.set_target_number_of_atoms(int(64))
            box.set_density(density, epsilon=0.4, dv=250, deform=False)
            box.make_initial_config(min_dist=2.0)

            # Generate LAMMPS input file
            # Units are in metal, so 1 fs timestep => tstep = 0.001

            box.set_lammps_params(print_reg=25)
            Datafile = os.path.join(WritePath, 'opls.data'.format(count))
            runfile = os.path.join(WritePath, 'opls.in'.format(count))

            lmpFilelist.append(runfile)

            box.write_lammps_datafile(Datafile)
            box.write_lammps_inputfile(EquilTempStart=5000.0, EquilTempStop=5000.0, SampleTemp=5000.0, tstep=0.0005,
                                       melt_steps=200000, equil_steps=25000000, sample_steps=2200000, nsims=16, datafile=Datafile, filename=runfile)
            #(25000, 1000000 )

        elif ffparam == 'TF':
            from .moltensalt_tosifumi_gen import simbox
            box = simbox()
            box.set_composition(composition, fraction)
            box.set_target_number_of_atoms(64)
            box.set_density(density, epsilon=0.4, dv=250, deform=False)
            box.make_initial_config(min_dist=2.0)
            box.set_lammps_params(print_reg=25)

            Datafile = os.path.join(WritePath, 'opls.data'.format(count))
            runfile = os.path.join(WritePath, 'opls.in'.format(count))

            lmpFilelist.append(runfile)

            box.write_lammps_datafile(Datafile)
            box.write_lammps_inputfile(EquilTempStart=3000.0, EquilTempStop=2200.0, SampleTemp=2200.0, tstep=0.0005,
                                       melt_steps=200000, equil_steps=25000000, sample_steps=2200000, nsims=16, datafile=Datafile, filename=runfile)

            if (composition == ['Li,Cl', 'K,Cl'] and fraction == [0.5, 0.5]):
                if not os.path.isdir(os.path.join(DataPath, 'deform')):
                    os.mkdir(os.path.join(DataPath, 'deform'))
                DeformPath = os.path.join(DataPath, 'deform')

                species = '{Li:0.0:K:0.0:Cl:0.0}'
                filename = 'species.dat'
                print(composition, fraction)
                if not os.path.isfile(os.path.join(DeformPath, filename)):
                    with open(os.path.join(DeformPath, filename), 'w') as f:
                        f.write(species)
                print('\nSpecies support for QUIP Written to : {}\n'.format(
                    os.path.join(DeformPath, filename)))

                runpath.append(DeformPath)
                box = simbox()
                box.set_composition(composition, fraction)
                box.set_target_number_of_atoms(64)
                box.set_density(density, epsilon=0.45, dv=250, deform=True)
                box.make_initial_config(min_dist=2.0)
                box.set_lammps_params(print_reg=25)
                Datafile = os.path.join(DeformPath, 'opls.data')
                runfile = os.path.join(DeformPath, 'opls.in')
                lmpFilelist.append(runfile)

                box.write_lammps_datafile(Datafile)
                box.write_lammps_inputfile(EquilTempStart=3000.0, EquilTempStop=2200.0, SampleTemp=2200.0, tstep=0.0005,
                                           melt_steps=200000, equil_steps=25000000, sample_steps=2200000, nsims=100, datafile=Datafile, filename=runfile)

    return runpath, lmpFilelist
