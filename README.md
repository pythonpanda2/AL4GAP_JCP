#  Active Learning  Workflow for Gaussian Approximation Potential (AL4GAP)
#### October 2022
This repository provides documentation for the  active learning workflow for [Gaussian approximation potentials](https://libatoms.github.io/GAP/index.html). The published article associated with this repository can be found [here](https://pubs.aip.org/aip/jcp/article/159/2/024802/2901782/AL4GAP-Active-learning-workflow-for-generating-DFT).

![](https://github.com/vswoo/AL4GAP/blob/main/AL4GAP_workflow.jpg)

The workflow capabilities includes: (1) setting up user-defined combinatorial chemical spaces of charge neutral mixtures of arbitrary molten mixtures spanning^ **11 cations (Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba and two heavy species, Nd and Th) and 4 anions (F, Cl, Br and I)**, (2) configurational sampling using low-cost empirical parameterizations, (3) ensemble active learning for down-selecting configurational samples for single point density functional theory calculations at the level of strongly constrained and appropriately normed (SCAN) exchange-correlation functional, and (4) Bayesian optimization for hyperparameter tuning of two-body and many-body GAP models*. 

**^The systems covered in the preprint explored  binary salt mixture space. But the workflow can handle more than two salt mixtures and can be any arbitrary charge neutral combination drawn from the supported element types**.


## Prerequisites
This workflow uses miniconda3 and Python 3.7. For Bebop users, Anaconda can be loaded with the command:
```bash 
$ module load anaconda3
``` 
Additionally, the LAMMPS molecular dynamics simulator is used for configuration sampling. A basic LAMMPS executable appropriate for your machine is required. Visit the [LAMMPS Documentation]( https://docs.lammps.org/Install.html) for more information on the installation.
## SmartSim Prerequisites:
Please refer to [SmartSim documentation](https://www.craylabs.org/docs/overview.html) for more information. The base prerequisites to install SmartSim and SmartRedis are:
-	Python 3.7-3.9
-	Pip
-	Cmake 3.13.x (or later)
-	C compiler
-	C++ compiler
-	GNU Make > 4.0
-	Git
-	git-lfs
Note that most developer systems already have many of these packages installed, but check that the package versions comply with the above prerequisites.

Next, use the following terminal commands to create the conda environment (“al4gap”) based on the repository requirements:
```bash
git clone https://github.com/vswoo/AL4GAP.git
```
```bash
cd AL4GAP
conda env create -f env.yml
cd MSMOD
cp -r AL4GAP  /your/path/al4gap/lib/python3.7/site-packages/
conda activate /your/path
```
## Install SmartSim for CPU
SmartSim and SmartRedis are installed with the env.yml file, however the SmartSim build requires git-lfs to be installed before the build process.
```bash
conda install git-lfs
smart clean
smart build --device cpu
```
## For Bebop Users
```text
Modules loaded in Bebop for Environment Build:
1) StdEnv 
2) cmake/3.20.3-vedypwm 
3) gcc/8.2.0-xhxgy33
4) gmake/4.2.1-expbj35

Modules loaded in Bebop for Running Experiment (loaded in driver.py):
1) StdEnv   
2) intel/cluster.2018.3
4) gcc/7.1.0
5) gsl/2.4
6) lammps/12Mar19

Check which modules are available with:
$ module avail
```
## Simple AL4GAP Tutorial 
To showcase the active learning framework, a simple example is provided in the Jupyter notebook, "tutorial.ipynb”. 
If necessary, the conda env can be used as kernel for the notebook using the following command:
```bash
jupyter kernelspec list
python -m ipykernel install --user --name=al4gap
```
Now, we will run a simple active learning example for the CaCl2-NdCl3 molten salt composition:
```
cd Notebook/
unzip To_QUIP.extxyz.zip
export PATH=/your/path/al4gap/lib/python3.7/site-packages/:$PATH
jupyter-notebook tutorial.ipynb
```
## Running AL4GAP
-	This script adopts the AL4GAP workflow and uses SmartSim to launch the workload on Bebop worker nodes with Slurm WLM. 
-	SmartSim ensemble feature is used to launch and run multiple experiments simultaneously as a batch.
-	The allocations are given as (# Compositions + # DB Node) nodes.
- The driver.py script is currently set to run the "densities.csv" file for 3 MS compositions, this can be easily changed to "example.csv" to run 1 composition. Make sure compute resources are adjusted accordingly.

```bash
sbatch submit_driver.sh
```
## Bayesian Optimization of GAP Model Hyperparameters
DFT-SCAN calculations are performed on the AL4GAP samples best configutations from across the compositions. Here we will utilize DFT-SCAN data for   **KCl-ThCl<sub>4</sub>** melt compositions (listed in Appendix A "Composition space for five molten salt mixture
chemistry" of the article) to illustrate a Bayesian Optimization (BO) of a **2B+SOAP** GAP Model hyperparameters.  The BO script is implemted as [BayesOpt_SOAP.py](https://github.com/vswoo/AL4GAP/blob/main/HyperparameterOptimization/BayesOpt_SOAP.py). The following paramter search space is defined inside this script:

```
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


```
The full list of GAP hyperparameters can be read using the link [here](https://libatoms.github.io/GAP/gap_fit.html#command-line-example). The BO script is launched on a single HPC node as shown below

```
cd HyperparameterOptimization/

sbatch bdw.sh
```

The optimized result for this run can be seen at the end of the [BO-SOAP.out](https://github.com/vswoo/AL4GAP/blob/main/HyperparameterOptimization/BO-SOAP.out) output file and also written to [hyperparam_quip.json](https://github.com/vswoo/AL4GAP/blob/main/HyperparameterOptimization/hyperparam_quip.json) file is shown below.

```
{"cutoff": 6.1104489231575405, 
"delta2b": 11.112350598173125, 
"delta": 0.8938947619284, 
"n_sparse": 1500.0, 
"n_sparse2b": 25.0,
"lmax": 4.0, 
"nmax": 9.0,
"MAE": 0.458909397734845}
```

### Additional notes
 Retraining on metadynamics is covered in the published article. Retraining on a few configurations  sampled at different volume using the partially trained GAP model is also helpful. 


## How to cite ?
If you are using the AL4GAP workflow  in your research, please cite us as
```
@article{guo_woo_andersson_hoyt_williamson_foster_benmore_jackson_sivaraman_2023,
    author = {Guo, Jicheng and Woo, Vanessa and Andersson, David A. and Hoyt, Nathaniel and Williamson, Mark and Foster, Ian and Benmore, Chris and Jackson, Nicholas E. and Sivaraman, Ganesh},
    title = "{AL4GAP: Active learning workflow for generating DFT-SCAN accurate machine-learning potentials for combinatorial molten salt mixtures}",
    journal = {The Journal of Chemical Physics},
    volume = {159},
    number = {2},
    pages = {024802},
    year = {2023},
    month = {07},
    issn = {0021-9606},
    doi = {10.1063/5.0153021},
    url = {https://doi.org/10.1063/5.0153021},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0153021/18037065/024802\_1\_5.0153021.pdf},
}
```

[![DOI](https://zenodo.org/badge/623237723.svg)](https://zenodo.org/badge/latestdoi/623237723)



## Contributions

- AL4GAP was conceptualized and implemented by [Ganesh Sivaraman](https://github.com/pythonpanda2). 

- [Ganesh Sivaraman](https://github.com/pythonpanda2) wrote the core components of AL4GAP with collaboration/ contribution from [Prof. Nicholas Jackson](https://github.com/TheJacksonLab).

- [Vanessa Woo](https://github.com/vswoo) implemented the AL4GAP ensemble driver script, added bug fix to atom packing method, added tutorials, created the workflow image, and wrote the GitHub documentations with input from GS.


## Acknowledgments
This material is based upon work supported by Laboratory Directed Research and Development (LDRD-CLS-1-630) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357. This research was in portion supported by ExaLearn Co-design Center of the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration. Portions of this work were sponsored by the U.S. Department of Energy, Office of Nuclear Energy’s Material Recovery and Wasteform Development Program under contract DE-AC02-06CH11357. We gratefully acknowledge the computing resources provided on Bebop; a high-performance computing cluster operated by the Laboratory Computing Resource Center at Argonne National Laboratory. This research used resources of the Argonne Leadership Computing Facility, a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.
