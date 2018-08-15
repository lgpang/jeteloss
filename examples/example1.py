#!/usr/bin/env python
'''
One simple example to use RAA2Eloss 

(1) In this example, we load pt spectra for proton+proton collisions
at sqrts=2760 GeV (from Pythia simulations, provided by default).

(2) Then we load RAA data which is stored in data/RAA_pbpb2760.txt,
The RAA data file should have 4 columns,

        column 1, a list of transverse momenta pt, in units [GeV]
        column 2, the error in pt, in units [GeV], not used
        column 3, a list of RAA y values
        column 4, the experimental error of RAA y values

lines start with '#' are neglected (treated as comments).

(3) Create RAA2Eloss() object

(4) Train the mcmc model

(5) Save the model parameters to file

(6) Make plots for mean pt loss and pt loss distribution

Example:
    >>> import numpy as np
    >>> from jeteloss import PythiaPP, RAA2Eloss
    >>> pp_x, pp_y = PythiaPP(sqrts_in_gev = 2760)
    >>> raa_fname = "RAA_pbpb2760.dat"
    >>> eloss = RAA2Eloss(raa_fname, pp_x, pp_y)
    >>> eloss.train(steps=20000, burn=10000, thin=2)
    >>> eloss.save_results()
    >>> jet_pt = np.linspace(1, 400, 200)
    >>> eloss.plot_mean_ptloss(jet_pt)
    >>> eloss.plot_pt_loss_dist(jet_pt=110)
'''

import os
import numpy as np
from jeteloss import PythiaPP, RAA2Eloss

pp_x, pp_y = PythiaPP(sqrts_in_gev = 2760)
raa_fname = os.path.join('.', "RAA_2760.txt")
eloss = RAA2Eloss(raa_fname, pp_x, pp_y)
eloss.train(steps=20000, burn=10000, thin=2)
eloss.save_results()
pt = np.linspace(1, 200, 100)
eloss.plot_RAA(save_name="RAA_cmp.png")
eloss.plot_mean_ptloss(pt)
eloss.plot_pt_loss_dist(jet_pt=110)
eloss.plot_pymc_summary()
