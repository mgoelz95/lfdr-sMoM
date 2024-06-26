#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define  the parameters for scenarios and sensor configurations.

@author: Martin Goelz
"""
import os
import sys

import _pickle as pickle

import numpy as np

import pandas as pd

def get_par_mbm_em(dat_path, par="stan"):
    """Get parameters for MBM expectation maximization.

    Parameters
    ----------
    dat_path : str
        The path to where the data is stored.
    par : str
        The name of the parametrization to be used. The default is "stan".

    Returns
    -------
    Returns
    -------
    K_cnds : numpy array
        The candidate model orders.
    n_ran_init : int
        The number of random initializations of the parameter values.
    mom_n_tr : int
        How many times random vectors are partitioned.
    cvg_thr : float
        The convergence threshold for EM.
    """
    try:
        with open(os.path.join(
                dat_path, "..", "mbm-em_par_") + par + '.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return (loaded[0], loaded[1], loaded[2])
    except FileNotFoundError:
        if par == "stan":
            # The candidate model orders
            K_cnds = np.arange(1, 21, 1)
            # The number of random initializations per candidate
            n_ran_init = 5
            # The convergence threshold for EM
            cvg_thr = 1e-5
            with open(
                os.path.join(dat_path, "..", "mbm-em_par_") + par + '.pkl',
                'wb') as output:
                pickle.dump([K_cnds, n_ran_init, cvg_thr],
                            output, -1)
        else:
            print('This parametrization type has not been implemented for MBM'
            ' EM (yet)!')
            sys.exit()
        return K_cnds, n_ran_init, cvg_thr


def get_par_fd_scen(fd_scen, dat_path):
    """
    Returns the parameters for the given field scenario.

    Parameters
    ----------
    fd_scen : str
        The scenario name.
    dat_path : str
        The path to where the data is stored.

    Returns
    -------
    tuple
        The field dimensions in (y, x).
    int
        The number of MC runs.
    int
        The number of observation samples per grid point.
    int
        The number of active sources in the field.
    float
        The desired proprotion of grid points where H0 holds (between 0 and 1).
    logical
        Whether there is shadow fading or not to be simulated.
    str
        The desired propagation environment. Urban or suburban.

    """
    try:
        with open(os.path.join(
                dat_path, fd_scen) + '_par.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return (loaded[0], loaded[1], loaded[2], loaded[3], loaded[8],
                loaded[9], loaded[10])
    except FileNotFoundError:
        if fd_scen == 'scA_TSIPN' or fd_scen == 'scA_CISS':
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 5
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False
            # Ground truth desired null proportion of pixels
            pi0_des = .2

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "suburb"  # suffix dependning on whether urban or
            # suburban is considered
        elif fd_scen == 'scA_ICASSP':
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 2
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .9

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "suburb"  # suffix dependning on whether urban or
            # suburban is considered
        elif (fd_scen == 'scB_TSIPN' or fd_scen == 'scB_ICASSP'
              or fd_scen == 'scB_CISS'):
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 8
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .6

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "suburb"  # suffix dependning on whether urban or
            # suburban is considered
        elif fd_scen == 'scC_TSPIN':
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 2
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .9

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "suburb"  # suffix dependning on whether urban or
            # suburban is considered
        elif fd_scen == 'scC_CISS':
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 2
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .9

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "urb"  # suffix dependning on whether urban or
            # suburban is considered
        elif fd_scen == 'scC_ICASSP':
            fd_dim = (100, 100)
            n_MC = 10
            n_sam = 1024
            n_src = 1
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .6

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "urb"  # suffix dependning on whether urban or suburban
            # is considered
        elif fd_scen == 'sc_3MT':
            fd_dim = (100, 100)
            n_MC = 20
            n_sam = 1024
            n_src = 2
            ran_cen = True
            ran_rad = False
            ran_pre = False
            add_tra = False

            # Ground truth desired null proportion of pixels
            pi0_des = .6

            # Do we use a field with shadow fading?
            sha_fa = True
            prop_env = "urb"  # suffix dependning on whether urban or suburban
            # is considered
        else:
            try:
                custom_vls = pd.read_pickle(
                    os.path.join(dat_path, '..', fd_scen + '.pkl'))
            except FileNotFoundError:
                print("This scenario name is undefined!")
                sys.exit()
            print("Loading custom p-values...", end="")
            try:
                fd_dim = custom_vls['fd_dim'][0]
            except KeyError:
                fd_dim = np.nan
            n_MC = custom_vls['p'][0].shape[0]
            print(" complete!")
            return fd_dim, n_MC, np.nan, np.nan, np.nan, np.nan, np.nan

        with open(os.path.join(
                dat_path, fd_scen) + '_par.pkl', 'wb') as output:
            pickle.dump([fd_dim, n_MC, n_sam, n_src, ran_cen, ran_rad, ran_pre,
                         add_tra, pi0_des, sha_fa, prop_env], output, -1)
    return fd_dim, n_MC, n_sam, n_src, pi0_des, sha_fa, prop_env


def get_par_quan(n_bits, dat_path, lam=1):
    """Load the quantization parameters.

    The user has to be define the number of quantization bits, the bin widths
    are:
        - i/\sum(1:I)
    Here, i denotes a bin index, i \in [I] and I = 2^n_bits is the number of
    bins.

    Parameters
    ----------
    n_bits : int
        The number of quantization bits. Recommended: No more than 8 because 8
        bits usually form one block (depending on transmission protocol)
    lam : int, optional
        The upper end of the quantization interval. The default is 1, i.e., the
        entire p-val range is quantized. Loweing it potentially reduces the num
        of needed bits for quantization, at the cost of having no resolution at
        all beyond lam.

    Returns
    -------
    q_borders : numpy array
        The borders of all bins.
    q_width : numpy array
        The width of all arrays.
    q_centers : numpy array
        The center of all arrays.
    """
    try:
        with open(os.path.join(dat_path, '..',
        f'quan_{n_bits}Bit_sensoring_at_{lam}') + '_par.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return loaded[0], loaded[1], loaded[2]
    except FileNotFoundError:
        n_bins = 2 ** n_bits

        q_width_no_lam = np.arange(n_bins) + 1
        q_width_no_lam = q_width_no_lam / np.sum(q_width_no_lam) * lam
        if lam < 1:
            q_width = np.concatenate((q_width_no_lam, np.array([1 - lam])))
        else:
            q_width = q_width_no_lam
        q_borders = np.concatenate((
            np.array([0]), np.cumsum(q_width)))
        q_centers = q_width/2 + q_borders[:-1]
        with open(os.path.join(dat_path, '..',
        f'quan_{n_bits}Bit_sensoring_at_{lam}') + '_par.pkl', 'wb') as output:
            pickle.dump([q_borders, q_width, q_centers], output, -1)
    return (q_borders, q_width, q_centers)

def get_par_sen_cfg(sen_cfg, dat_path):
    """
    Returns the parameters for the given sensor configuration.

    Parameters
    ----------
    sen_cfg : str
        The sensor configuration.
    dat_path : str
        The path where the data is stored.

    Returns
    -------
    numpy array
        The number of sensors of each type.
    numpy array
        The number of observation samples per sensor type.
    logical
        True if sensor locations shall vary with MC runs.
    logical
        If sensor are to be distributed homogeneously.

    """
    try:
        with open(os.path.join(
                dat_path, sen_cfg, sen_cfg) + '_par.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return (loaded[0], loaded[1], loaded[2], loaded[3])
    except FileNotFoundError:
        if sen_cfg == 'cfg1_TSIPN':
            n_sen = np.array([10000])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == 'cfg2_TSIPN':
            n_sen = np.array([300])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == 'cfg3_TSIPN':
            n_sen = np.array([270, 30])
            n_sam_per_sen = np.array([256, 1024])
            var_sen_loc = np.array([True, True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True, True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == 'stan_300':
            # Used for first column in Table 2 of [Goelz2022ICASSP], Fig 2 of
            # [Goelz2022ICASSP] and Fig 4 of [Goelz2022CISS]
            n_sen = np.array([300])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == '3MT_1500':
            # USED FOR 3MT!
            n_sen = np.array([1500])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == 'stan_1000':
            # Used for second column in Table 2 of [Goelz2022ICASSP] and Fig 3
            # of [Goelz2022ICASSP]
            n_sen = np.array([1000])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        elif sen_cfg == 'stan_3000':
            # Used for third column in Table 2 of [Goelz2022ICASSP] and Fig 5
            # of [Goelz2022CISS]
            n_sen = np.array([3000])
            n_sam_per_sen = np.array([256])
            var_sen_loc = np.array([True])  # If true, sensor locations varying
            # with MC run
            sen_hom = np.array([True])  # If true, sensors are homogeneously
            # distributed across field
        else:
            print("Loading custom sensor configuration...", end="")
            custom_vls = pd.read_pickle(
                os.path.join(dat_path, '..', sen_cfg + '.pkl'))
            n_sen = custom_vls['p'][0].shape[1]
            return n_sen, np.nan, np.nan, np.nan
            print(" complete!")
        with open(os.path.join(
                dat_path, sen_cfg, sen_cfg) + '_par.pkl', 'wb') as output:
            pickle.dump([n_sen, n_sam_per_sen, var_sen_loc, sen_hom],
                        output, -1)
    return n_sen, n_sam_per_sen, var_sen_loc, sen_hom

def get_par_spa_var(dat_path, par="stan"):
    """
    Get parameters for spatially varying null probability.

    Parameters
    ----------
    dat_path : str
        The path to where the data is stored.
    par : str
        The name of the parametrization to be used. The default is "stan".

    Returns
    -------
    """
    try:
        with open(os.path.join(
                dat_path, "..", "spa_var_par_") + par + '.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return (loaded[0], loaded[1], loaded[2], loaded[3])
    except FileNotFoundError:
        if par == "stan":
            bw_grid = np.arange(.01, 20, .5)
            ker_vec = ['epa', 'gauss', 'exp', 'lin', 'cos']
            laws_sthr = np.array([.9])  # As recommended in [Cai2021]
            pi0_max = .99
            with open(os.path.join(dat_path, "..", "spa_var_par_")
                      + par + '.pkl', 'wb') as output:
                pickle.dump([bw_grid, ker_vec, laws_sthr, pi0_max],
                            output, -1)
        else:
            print("This parametrization for the spatially varying prior has "
            "not been implemented (yet)")
            sys.exit()
        return bw_grid, ker_vec, laws_sthr, pi0_max

def get_par_smom(dat_path, par="stan"):
    """
    Get parameters for spectral method of moments.

    Parameters
    ----------
    dat_path : str
        The path to where the data is stored.
    par : str
        The name of the parametrization to be used. The default is "stan".

    Returns
    -------
    mom_k : numpy array
        The grid for the number of multivariate components in the mixture
        model.
    mom_d : numpy array
        The grid for sizes of the multivariate vectors.
    mom_n_tr : int
        How many times random vectors are partitioned.
    reps_eta : int
        How many Gaussian vectors are generated for the spectral ethod of
        moments.
    dis_msr : str
        The distance measure used to evaluate a good fit.
        Options: Wasserstein distance (recommended), Jensen-Shannon divergence,
        Kolmogorov-Smirnof distance, Kullback-Leibler divergence.

    """
    try:
        with open(os.path.join(
                dat_path, "..", "smom_par_") + par + '.pkl', 'rb') as input:
            loaded = pickle.load(input)
        return (loaded[0], loaded[1], loaded[2], loaded[3], loaded[4])
    except FileNotFoundError:
        if par == "stan":
            # Must have integer square root for spatial partition
            mom_d = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16])
            mom_k = np.array([1, 2, 3, 4, 5, 6])  # Number of multivariate
            # components
            mom_n_tr = 10  # Number of random partitions of the data into
            # multivariate p-value vectors
            reps_eta = 10  # Number of trials for the spectral method of
            # moments with random eta vectors.
            dis_msr = 'was'  # Distance measure for evaluation of goodness of
            # fit.
            # Options: was, js, kl, ks (see paper for mor edetails)
            with open(os.path.join(dat_path, "..", "smom_par_")
                      + par + '.pkl', 'wb') as output:
                pickle.dump([mom_k, mom_d, mom_n_tr, reps_eta, dis_msr],
                            output, -1)
        else:
            print('This parametrization for smom has not been implemented '
                '(yet)')
            sys.exit()
        return mom_k, mom_d, mom_n_tr, reps_eta, dis_msr
