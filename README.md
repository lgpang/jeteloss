# Data driven extraction of jet energy loss distributions in heavy ion collisions

## Introduction

This python package is a simple tool to extract mean pt loss $P(\Delta p_T)$,
and the mean pt loss as a function of jet pt -- $\langle\Delta p_T\rangle(p_T)$,
from the experimental single jet RAA for AA collisions at a specific beam energy 
(with pt spectra in proton+proton collisions at the same beam energy) or the single hadron/gamma hadron
pt spectra (without pt spectra in proton+proton collisions).

Example:
    >>> from jeteloss import PythiaPP, RAA2Eloss
    >>> pp_x, pp_y = PythiaPP(sqrts_in_gev = 2760)
    >>> raa_fname = "RAA_2760.txt"
    >>> eloss = RAA2Eloss(raa_fname, pp_x, pp_y)
    >>> eloss.train()
    >>> eloss.save_results()
    >>> eloss.plot_mean_ptloss()
    >>> eloss.plot_pt_loss_dist()

## Citation

If you have used this package to produce results for presentation/publications,
please cite the following two papers, from where one can find the detailed information of 
the underlying physics.


## Installation

### Method 1: using pip
Step 1: 
> pip install jeteloss

Step 2:
> git clone git@github.com:lgpang/jeteloss.git

Step 3:
> cd jeteloss/examples

> python example1.py

### Method 2: install from local directory
Step 1: download the code from github
> git clone git@github.com:lgpang/jeteloss.git

Step 2: install jeteloss and dependences
> cd jeteloss

> python setup.py install

Step 3: run example code
> cd examples

> python example1.py

### Method 3: using anaconda

Step 1: To create one clean python virtual environment 
> conda create -n test_jeteloss python=3.6

Step 2: To activate this environment, use:
> source activate test_jeteloss

Step 3: Install jeteloss module and its dependences
> pip install jeteloss

Step 4: Run the example code downloaded using:
> git clone git@github.com:lgpang/jeteloss.git

> cd jeteloss/examples

> python example1.py

Step 5: To deactivate an active environment, use:
> source deactivate

Step 6: Clean up
To see how many environments do you have, use:
> conda env list

To remove one environment, use:
> conda remove --name test_jeteloss --all

