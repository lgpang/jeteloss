#!/usr/bin/env python
# -*- coding: utf-8 -*-
# by Long-Gang Pang, Yayun He and Xin-Nian Wang
# bug report: lgpang@qq.com
__AUTHORS__ = "Long-Gang Pang, Ya-Yun He and Xin-Nian Wang"
__REPORT__ = "lgpang@qq.com"
'''
Data-driven extraction of jet energy loss distributions in heavy-ion collisions

This module defines a data-driven extractor for jet energy loss in heavy-ion collisions.
Using this module, experimentalists can extract the distribution of jet energy loss -- P(Delta p_T),
and the mean pt loss as a function of jet pt -- <Delta p_T>(p_T), as long as they provide
the experimental single jet RAA for AA collisions at a specific beam energy 
(with pt spectra in proton+proton collisions at the same beam energy) or the single hadron/gamma hadron
pt spectra (without pt spectra in proton+proton collisions).

The pt spectra of proton+proton collisions at $\sqrt{s_{NN}}$= 200 GeV, 2.76 TeV and 5.02 TeV,
are delivered by default, using Pythia8 simulations.

For detailed theoretical discription please refer to: arxiv_id or bibtex info

Example:
    >>> from jeteloss import PythiaPP, RAA2Eloss
    >>> pp_x, pp_y = PythiaPP(sqrts_in_gev = 2760)
    >>> raa_fname = "./data/RAA_2760.txt"
    >>> eloss = RAA2Eloss(raa_fname, pp_x, pp_y)
    >>> eloss.train()
    >>> eloss.save_results()
    >>> eloss.plot_mean_ptloss()
    >>> eloss.plot_pt_loss_dist()
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
from pymc import Uniform, Normal, deterministic
from scipy.special import gamma
from scipy.interpolate import interp1d
import h5py
import pandas as pd


# the 15 point Gauss–Laguerre quadrature, which is used for 
# Delta_pt integration from 0 to infinity
gala15x = np.array([0.093307812017, 0.492691740302,
    1.215595412071,         2.269949526204,
    3.667622721751,         5.425336627414,
    7.565916226613,        10.120228568019,
    13.130282482176,        16.654407708330,
    20.776478899449,        25.623894226729,
    31.407519169754,        38.530683306486,
    48.026085572686])

gala15w = np.array([0.239578170311, 0.560100842793,
    0.887008262919,         1.22366440215,
    1.57444872163,          1.94475197653,
    2.34150205664,          2.77404192683,
    3.25564334640,          3.80631171423,
    4.45847775384,          5.27001778443,
    6.35956346973,          8.03178763212,
    11.5277721009])

gala30x = np.array([
0.047407180540804851462, 0.249923916753160223994,
0.614833454392768284613, 1.143195825666100798284,
1.836454554622572291486, 2.69652187455721519578,
3.72581450777950894932, 4.92729376584988240966,
6.304515590965074522826, 7.861693293370260468756,
9.603775985479262079849, 11.5365465979561397008,
13.6667446930642362949, 16.00222118898106625462,
18.55213484014315012403, 21.32720432178312892793,
24.34003576453269340098, 27.60555479678096102758,
31.14158670111123581829, 34.96965200824906954359,
39.11608494906788912192, 43.61365290848482780666,
48.50398616380420042732, 53.84138540650750561747,
59.69912185923549547712, 66.18061779443848965169,
73.44123859555988223951, 81.73681050672768572223,
91.5564665225368382555, 104.1575244310588945118
])

gala30w = np.array([
0.121677895367261782329, 0.283556882734938525493,
0.446432426678773424704, 0.610532130075812839931,
0.77630347862220587813, 0.944233288641719426021,
1.11484470167521153566, 1.288705432832675646608,
1.466439137624214862, 1.64873949785431911827,
1.83638767787041103674, 2.03027425167718247131,
2.23142718244322451682, 2.44104811309112149776,
2.66056024337508997273, 2.89167264137737623469,
3.13646833382438459, 3.3975275957906408941,
3.67810472956067256012, 3.9823886239009637486,
4.315899244899456113557, 4.686114009126298021,
5.1035026514834184093, 5.58333298168755125349,
6.1490444086574353235, 6.8391305457947582711,
7.7229478770061872416, 8.9437424683710389523,
10.87187293837791147282, 15.026111628122932986
])

def gamma_dist(x, alpha, beta):
    ''' gamma distribution whose mean = alpha/beta and var=alpha/beta**2

        Args:
            alpha: shape parameter
            beta: inverse scale parameter
    '''
    return beta**(alpha) / gamma(alpha) * np.exp(-beta*x) * x**(alpha-1)


def PythiaPP(sqrts_in_gev=2760):
    ''' read saved pt spectra of p+p collision

        Args:
            sqrts_in_gev (int): beam energy in unit [GeV]

        Return:
            (pt, spectra) where both pt and spectra are 1D numpy arrays
    '''
    cwd, cwf = os.path.split(__file__)
    dat = np.loadtxt(os.path.join(cwd, 'data/pp_%s.txt'%sqrts_in_gev))
    return dat[:, 0], dat[:, 1]


class RAA2Eloss(object):
    '''A data driven jet eloss extractor
    
    Compute the distribution of the jet pt loss and
    the mean pt loss as a function of jet pt '''
    def __init__(self, RAA_fname, pp_pt=None, pp_spectra=None):
        '''Args:
               RAA_fname: a data file that stores the RAA data, with 4 columns,
                   column 1, a list of transverse momenta pt, in units [GeV]
                   column 2, the error in pt, in units [GeV], not used
                   column 3, a list of RAA y values
                   column 4, the experimental error of RAA y values
                   lines start with '#' are neglected (treated as comments).
                   If pp_pt=None, pp_spectra=None, RAA data can be the single
                   hadron spectra assuming pp_spectra = 1 for all pts.
               pp_pt (1d numpy.array): the transverse momentum of the pt spectra in p+p collisions
               pp_spetra (1d numpy.array): the pt spetra y values in p+p collisions with the same
                   beam energy as in the A+A collisions for RAA

           Returns:
               One data driven Bayesian model that can be trained using MCMC,
               and visualized using matplotlib.
        '''
        self.with_ppdata = False
        if not (pp_pt is None or pp_spectra is None):
            self.pp_fit = interp1d(pp_pt, pp_spectra, fill_value="extrapolate")
            self.with_ppdata = True
        RAA_data = np.loadtxt(RAA_fname)
        self.RAA_x = RAA_data[:, 0]
        self.RAA_xerr = RAA_data[:, 1]
        self.RAA_y = RAA_data[:, 2]
        self.RAA_yerr = RAA_data[:, 3]

    def __constrain_params__(self, variable='a', n_sigma=1.0):
        '''constrain parameters to one sigma of RAA fitting'''
        model = self.mdl_
        mu = model.trace('mu')[...]
        raa_mean = np.array(mu).mean(axis=0)
        raa_sd = np.array(mu).std(axis=0)
        value = model.trace(variable)[...]
        select_mask = (mu > raa_mean - n_sigma*raa_sd)*(mu < raa_mean + n_sigma*raa_sd)
        select_mask = np.all(select_mask, axis=1)
        return value[select_mask]

    def mean_ptloss(self, pt, b, c):
        '''Parameterized mean pt loss as a function of pt

        Args:
            pt (1d np.array): array of transverse momentum coordinates
            b (float): controls the magnitude of the <Delta pt>
            c (float): controls the slop of the <Delta pt>

        Return:
            <Delta pt>(pt) (1d np.array): the mean pt loss
        '''
        return b*pt**c*np.log(pt)

    def __model__(self, RAA_x, RAA_y, RAA_xerr, RAA_yerr):
        '''RAA model for pymc '''
        a = Uniform('a', lower=0, upper=10)
        b = Uniform('b', lower=0, upper=10)
        c = Uniform('c', lower=0, upper=1)
        @deterministic(plot=False)
        def mu(a=a, b=b, c=c):  
            '''compute the RAA from convolution: 
              int_d(Delta_pt) P(Delta_pT) * sigma(pt+Delta_pt) / sigma(pt)
              '''
            intg_res = np.zeros_like(RAA_x)
            for i, x in enumerate(gala30x):
                # integral DeltaPt from 0 to infinity
                scale_fct = RAA_x / gala30x[-1]
                x = x * scale_fct
                shifted_pt = RAA_x + x
                mean_dpt = self.mean_ptloss(shifted_pt, b, c)
                alpha = a
                beta = a / mean_dpt
                pdpt = gamma_dist(x, alpha, beta)
                if self.with_ppdata:
                    intg_res += scale_fct*gala30w[i]*pdpt*self.pp_fit(shifted_pt)
                else:
                    intg_res += scale_fct*gala30w[i]*pdpt

            if self.with_ppdata:
                return intg_res / self.pp_fit(RAA_x)
            else:
                return intg_res
        likelihood_y = Normal('likelihood_y', mu=mu, tau=1/(RAA_yerr)**2, observed=True, value=RAA_y)
        return locals()

    def train(self, steps=20000, burn=10000, thin=10):
        '''train the Markov Chain Monte Carlo to estimate the parameters (b, c) in 
        mean pt loss and (a,) in pt loss distribution.

        Args: 
            steps (int): total number of Metropolis-Hastings random walk steps in the parameter space
                        default: 200000 steps
            burn (int): number of steps that are not used for final statistics; the estimated number
                        of steps for the MCMC to converge, default: 100000 steps
            thin (int): skipping number of steps to avoid auto-correlation, for final analysis.
                        The auto-correlation comes from Markov Chain (the next step in parameter space
                        is proposed using normal distribution whose mean is the current step).

        Updated:
            self.mdl_: MCMC model, which can be accessed by function self.get_mcmc_model()
            self.a (1d np.array): the trace of 'a' in the parameter space,
            self.b (1d np.array): the trace of 'b' in the parameter space,
            self.c (1d np.array): the trace of 'c' in the parameter space,
        '''
        raa_map = pm.MAP(self.__model__(self.RAA_x, self.RAA_y, self.RAA_xerr, self.RAA_yerr))
        raa_map.fit(iterlim=20000, tol=0.01)
        self.mdl_ = pm.MCMC(raa_map.variables)
        self.mdl_.sample(steps, burn, thin, verbose=0)
        self.a = self.mdl_.trace('a')[...]
        self.b = self.mdl_.trace('b')[...]
        self.c = self.mdl_.trace('c')[...]

    def trained_mcmc_model(self):
        ''' Retrun the trained MCMC model for other usage such as 
        convergence diagonostics, sample new traces

        Returns:
            self.mdl_: the trained mcmc model
        Raise:
            Exception("Train the model first using train(steps, burn, thin)!")
            if self.mdl_ does not exist.
        '''
        if self.mdl_:
            return self.mdl_
        else:
            raise Exception("Train the model first using train(steps, burn, thin)!")

    def save_results(self, save_directory='results/', save_name="pbpb2760_cent010.h5"):
        '''save the results to one hdf5 file for further processing

        Args: 
            save_directory (string): directory to save the results
            save_name (string): filename with surfix .h5 or .hdf5

        Return:
            save the RAA, a, b, c to save_directory/save_name which can be accessed using:
            >>> import h5py
            >>> with h5py.File( save_directory/save_name ) as h5:
            >>>     RAA = h5["RAA"][...]
            >>>     a = h5["a"][...]
        '''
        RAA_ = self.mdl_.trace('mu')[...]
        a_ = self.mdl_.trace('a')[...]
        b_ = self.mdl_.trace('b')[...]
        c_ = self.mdl_.trace('c')[...]
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fout = os.path.join(save_directory, save_name)
        if os.path.exists(fout): os.remove(fout)
        with h5py.File(fout, 'w') as h5:
            h5.create_dataset("RAA", data=RAA_)
            h5.create_dataset("a", data=a_)
            h5.create_dataset("b", data=b_)
            h5.create_dataset("c", data=c_)

    def plot_RAA(self, save_directory='figs/', save_name='RAA.png', jupyter=False):
        '''Args:
               pt (float): GeV, the jet initial pt
               save_directory: fig saving directory
               save_name: image saving name, format can be png, pdf, jpg or eps
               jupyter: True to show the plot when using jupyter-notebook
        '''
        variable='mu'
        mu = self.mdl_.trace(variable)[...]
        fit_mean = np.array(mu).mean(axis=0)
        fit_sd = np.array(mu).std(axis=0)
        plt.plot(self.RAA_x, fit_mean)
        plt.fill_between(self.RAA_x, fit_mean-fit_sd, fit_mean+fit_sd,
                color='0.5', alpha=0.5, label="Bayesian")
        plt.errorbar(self.RAA_x, self.RAA_y, yerr=self.RAA_yerr, color='r',
                marker='.', ls='None', label='Experiment')
        plt.ylabel(r'$R_{AA}$')
        plt.xlabel(r'$p_T$ [GeV]')
        plt.legend(loc='upper left')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fout = os.path.join(save_directory, save_name)
        plt.savefig(fout)
        print("Plotting fitting RAA .vs. data %s"%fout)
        if not jupyter: plt.close()

    def plot_mean_ptloss(self, jet_pt=None, ntracks=100,
            save_directory='figs/', save_name='mean_pt_loss.png',
            jupyter=False):
        ''' plot the mean pt loss as a function of jet pt

        Args:
            jet_pt (1d np.array): GeV, the jet initial pt; 
                if is None, will set it to: np.linspace(0, 400, 100)
            ntracks (int): num of tracks to plot besides <mean_pt_loss>
            save_directory: fig saving directory
            save_name: image saving name, format can be png, pdf, jpg or eps
            jupyter: True to show the plot when using jupyter-notebook
        '''
        if jet_pt is None:
            jet_pt = np.linspace(0, 400, 100)

        b = self.__constrain_params__('b')
        c = self.__constrain_params__('c')
        nsampling = len(b)
        nskip = 0
        if ntracks > nsampling: 
            ntracks = nsampling
            nskip = 1
        else:
            nskip = int(nsampling / ntracks) + 1
        y = np.zeros_like(jet_pt)
        for eid in range(nsampling):
            popt = (b[eid], c[eid])
            y += self.mean_ptloss(jet_pt, *popt)
            if nsampling % nskip == 0:
                plt.plot(jet_pt, self.mean_ptloss(jet_pt, *popt), 'b-', alpha=0.01)
        plt.plot(jet_pt, y/nsampling, 'mo-', label='Bayesian Mean')
        plt.xlabel(r"$p_T$ [GeV]")
        plt.ylabel(r"$\langle \Delta p_T \rangle$ [GeV]")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fout = os.path.join(save_directory, save_name)
        plt.savefig(fout)
        print("Plotting mean pt loss as a function of jet_pt to %s"%fout)
        if not jupyter: plt.close()

    def plot_pt_loss_dist(self, jet_pt=110, bins=100, logy=True,
            save_directory='figs/', save_name='pt_loss_dist.png',
            jupyter=False):
        '''plot the pt loss distribution as a function of jet pt 

        Args:
            jet_pt (float): GeV, the jet initial pt
            bins (int): number of Delta_pt points
            logy (bool): if True, plot y in log scale
            save_directory: fig saving directory
            save_name: image saving name, format can be png, pdf, jpg or eps
            jupyter: True to show the plot when using jupyter-notebook
        '''
        a = self.__constrain_params__('a')
        b = self.__constrain_params__('b')
        c = self.__constrain_params__('c')
        a_mean = a.mean()
        b_mean = b.mean()
        c_mean = c.mean()
        mean_dpt = self.mean_ptloss(jet_pt, b_mean, c_mean)
        alpha = a_mean
        beta = a_mean / mean_dpt
        Delta_pt = np.linspace(0, jet_pt, bins)
        ptloss_dist = gamma_dist(Delta_pt, alpha, beta)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if logy:
            plt.semilogy(Delta_pt, ptloss_dist)
        else:
            plt.plot(Delta_pt, ptloss_dist)
        plt.xlabel(r"$\Delta p_T$ [GeV]")
        plt.ylabel(r"$P(\Delta p_T)$")
        plt.title(r"$p_T^{\rm jet}=%s$ [GeV]"%jet_pt)
        plt.xlim(0, jet_pt)
        fout = os.path.join(save_directory, save_name)
        plt.savefig(fout)
        print("Plotting pt loss distribution to %s"%fout)
        if not jupyter: plt.close()
            
    def plot_correlation(self, save_directory='figs/', save_name='correlation.png',
            jupyter=False):
        '''plot the pair correlation among the parameters.

        Args:
            save_directory: fig saving directory
            save_name: image saving name, format can be png, pdf, jpg or eps
            jupyter: True to show the plot when using jupyter-notebook
        '''
        a = self.__constrain_params__('a')
        b = self.__constrain_params__('b')
        c = self.__constrain_params__('c')
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        df = pd.DataFrame(data={"a": a, "b":b, "c":c})
        g = sns.PairGrid(df, despine=False,  diag_sharey=False)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_diag(sns.kdeplot, lw=3)
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        # new names for parameters
        # a -> \alpha
        # b -> \beta
        # c -> \gamma
        xlabels = ['$\\alpha$', '$\\beta$', '$\\gamma$']
        ylabels = ['$\\alpha$', '$\\beta$', '$\\gamma$']
        for i in range(len(xlabels)):
            g.axes[-1,i].xaxis.set_label_text(xlabels[i], fontsize=20)
            g.axes[-1,i].xaxis.set_tick_params(labelsize=20)
            g.axes[i,0].yaxis.set_label_text(ylabels[i], fontsize=20)
            g.axes[i,0].yaxis.set_tick_params(labelsize=20)

        g.axes[0,0].set_yticks([0.0, 0.5, 1.0])
        g.axes[0,0].set_yticklabels([0.0, 0.5, 1.0])
        g.axes[1,0].yaxis.labelpad = 18
        g.axes[1,0].set_yticks([0.0, 2.0, 4.0])
        g.axes[1,0].set_yticklabels([0, 2, 4])
        
        g.axes[-1,2].set_xticks([0.0, 0.2, 0.4])
        g.axes[-1,2].set_xticklabels([0, 0.2, 0.4])
        
        g.axes[2,0].set_xticks([0, 2, 4, 6, 8])
        g.axes[2,0].set_xticklabels([0, 2, 4, 6, 8])
        
        g.axes[2,1].set_xticks([0, 2, 4])
        g.axes[2,1].set_xticklabels([0, 2, 4])
        
        g.axes[0,0].set_ylim(0, 1.0)
        g.axes[1,0].set_ylim(0, 5.9)
        g.axes[2,0].set_ylim(-0.1, 0.5)
        g.axes[2,0].set_xlim(0, 9.9)
        g.axes[2,1].set_xlim(0, 5.9)
        g.axes[2,2].set_xlim(0, 0.49)

        g.axes[0, 1].axis('off')
        g.axes[0, 2].axis('off')
        g.axes[1, 2].axis('off')
        
        fout = os.path.join(save_directory, save_name)
        g.savefig(fout)
        print("Plotting correlation of the parameters to %s"%fout)
        if not jupyter: plt.close()

    def plot_pymc_summary(self, path='figs/'):
        '''using pymc to plot the distribution and auto-correlations of 
        variables in the model

        Args:
            path: saving directory'''
        pm.Matplot.plot(self.mdl_, path=path)

