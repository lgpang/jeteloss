# Data driven extraction of jet energy loss distributions in heavy ion collisions

## Introduction

This python package is a simple tool to extract mean pt loss and pt loss distribution
from measured jet RAA, using Bayesian analysis.

## Citation

If you have used this package to produce presentation/publications, please cite
the following two papers, from where one can find the detailed information of 
the underlying physics.


## Installation

### Method 1: using pip
Step 1: 
> pip install jeteloss

Step 2:
> git clone 

Step 3:
> cd jeteloss/examples

> python example1.py

### Method 2: install from local directory
Step 1: download the code from github
> git clone 

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
> git clone *

> cd jeteloss/examples

> python example1.py

Step 5: To deactivate an active environment, use:
> source deactivate

Step 6: Clean up
To see how many environments do you have, use:
> conda env list

To remove one environment, use:
> conda remove --name myenv --all


