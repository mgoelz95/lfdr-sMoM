#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this file to apply lfdr-sMoM and its extensions developped in the following
papers:
    - [Goelz2022TSIPN]:
            Gölz et al., "Multiple Hypothesis Testing Framework for Spatial
                          Signals"
            DOI: 10.1109/TSIPN.2022.3190735

    - [Goelz2022ICASSP]:
            Gölz et al,. "Improving Inference for Spatial Signals by Contextual
            False Discovery Rates"
            DOI: 10.1109/ICASSP43922.2022.9747596

    - [Goelz2022CISS]:
            Gölz et al., "Estimating Test Statistic Distributions for Multiple
            Hypothesis Testing in Sensor Networks"
            DOI: 10.1109/CISS53076.2022.9751186

Remark: For all pre-defined scenarios, the number of independent repititions (
    Monte Carlo runs) is set to 10, to prevent this example file from running
    for a very long time (depending on the number of sensors). Hence, the
    averaged performance measures may deviate from the ones in the papers due
    to numerical inaccuracies, since we used 200MC runs in the papers.
    If you'd like to change the number of MC runs for the pre-defined
    scenarios, adjust the value of n_MC for the respective scenario
    in file parameters.py.

PLEASE CITE THE CORRESPONDING PAPERS IF YOU USE THIS CODE IN YOUR WORK!

@author: Martin Gölz
"""

# =============================================================================
# Instructions on how to create a pickle file with custom spatial p-values:
# For spatial data, aka, p-values assumed to have been observed at given
# coordinates within an area of interest.
#   1) Make sure that the dimensions of the observed spatial field is loaded
#      into the workspace as a variable named fd_dim. The values in fd_dim
#      are the number of grid points in y and x direction.
#   2) Make sure that the p-values are loaded into the workspace as a variable
#      named p. p is a numpy array of floats with the dimensions nMC x n, where
#      nMC is the number of Monte Carlo runs and n is the number of sensors. If
#      you have only one Monte Carlo run, make sure that p still has two axes.
#   2) Make sure that the sensor coordinates are loaded into the workspace as a
#      variable named sen_crd. sen_crd is a numpy array of integers of the size
#      nMC x n x 2.
#   3) (Optional) If the ground truth is available, make sure that a numpy
#      array of booleans with name r_tru and dimension nMC x n is in the
#      workspace.
#      Whereever r_tru has an entry of 1, H1 is in place, and wherever it has
#      an entry of 0, H0 is in place.
#   4) Make sure that dat_path is in the workspace and indicates the path to
#      where data is stored and that this path exists.
#   5) Make sure you've got pandas imported (import pandas as pd)
#   6) Now save the above mentioned variables in a pickle file:
#        file_name = 'whatever_you_want_to_name_the_file'
#        custom_pval = pd.DataFrame(
#                    {"fd_dim": [fd_dim],
#                     "p": [p],
#                     "sen_cds": [sen_cds],
#                     "r_tru": [r_tru]  # This line is optional! Only if you
#                     # have a ground truth!
#                     })
#        custom_pval.to_pickle(
#               os.path.join(dat_path, '..', file_name + '.pkl'))
#   7) Change the value for FD_SCEN for FIELD_MODE == 'custom' to file_name.
#   8) Make sure that FIELD_MODE = 'custom'
#   9) Run this file and enjoy!
# =============================================================================
# Instructions on how to create a pickle file with custom p-values:
# For non-spatial data, aka, you simply have a number of p-values that do not
# exhibit any kind of known 2D structure, but are just a 1D vector of tests.
#   1) Make sure that the p-values are loaded into the workspace as a variable
#      named p. p is a numpy array of floats with the dimensions nMC x n, where
#      nMC is the number of Monte Carlo runs and n is the number of tests. If
#      you have only one Monte Carlo run, make sure that p still has two axes.
#   2) (Optional) If the ground truth is available, make sure that a numpy
#      array of booleans with name r_tru and dimension nMC x n is in the
#      workspace.
#      Wherever r_tru has an entry of 1, H1 is in place, and wherever it has
#      an entry of 0, H0 is in place.
#   4) Make sure that dat_path is in the workspace and indicates the path to
#      where data is stored and that this path exists.
#   5) Make sure you've got pandas imported (import pandas as pd)
#   6) Now save the above mentioned variables in a pickle file:
#        file_name = 'whatever_you_want_to_name_the_file'
#        custom_pval = pd.DataFrame(
#                    {"p": [p],
#                     "r_tru": [r_tru]  # This line is optional! Only if you
#                     # have a ground truth!
#                     })
#        custom_pval.to_pickle(
#               os.path.join(dat_path, '..', file_name + '.pkl'))
#   7) Change the value for FD_SCEN for FIELD_MODE=='non-spatial' to file_name.
#   8) Make sure that FIELD_MODE = 'non-spatial'
#   9) Run this file and enjoy!
# =============================================================================
# %% setup: import packages
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
# %% setup: what will become a package
import spatialmht.field_handling as fd_hdl
import spatialmht.lfdr_estimation as lfdr_est
import spatialmht.analysis as anal
import spatialmht.detectors as det

# %% setup: import custom files
import parameters as par
import paths

# %% setup: user input
SAVE_RESULTS = True

FIELD_MODE = "radio"  # Alternatives: radio, non-spatial, custom

if FIELD_MODE == 'radio':
    # Change the values here to use different scenarios/configurations
    FD_SCEN = "scB_ICASSP"  # name of simulated field scenario, see
    # parameters.py for all options and for defining your own.
    # Pre-defined options: sc{A, B, C}_{TSIPN, CISS, ICASSP}
    SEN_CFG = "stan_300"  # name of sensor configuration, see
    # parameters.py for all options and for defining your own.
    # Pre-defined options: cfg{1, 2, 3}_TSIPN, stan_{300, 1000, 3000}
elif FIELD_MODE == 'custom':
    FD_SCEN = 'example_custom_pvals'  # Provide here the name of the .pkl file
    # that contains the custom p-values! File must be located in dat_path
    SEN_CFG = FIELD_MODE  # Do not change this.
elif FIELD_MODE == "non-spatial":
    FD_SCEN = 'example_non-spatial_pvals'  # Provide here the name of the .pkl
    # file that contains the p-values! File must be located in dat_path
    SEN_CFG = FIELD_MODE  # Do not change this.

QUANTIZE = False  # If p-values are to be quantized

if QUANTIZE:
    N_QUAN_BITS = 4  # number of quantization bits
else:
    N_QUAN_BITS = None

SENSORING_LAM = 1  # Sensoring threshold, if 1 then no sensoring

DO_SPATIAL_PARTITION = True  # if true, simulates also p-value vectors formed
# by spatial partitioning, denoted as lfdr-sMoM_s in [Goelz2022TSIPN]
DO_VAR_SPATIAL_PRIOR = True  # if true, clfdr-sMoM from [Goelz2022ICASSP]is
# applied
# TURNING DO_EM = True MAY LEAD TO THE FILE RUNNING FOR A WHILE, DEPENDING ON
# THE SELECTED SCENARIO/CONFIGURATION AND YOUR HARDWARE
DO_EM = False  # if true, (c)lfdr-sMoM-EM from [Goelz2022CISS]is applied.
DO_TRUE = True  # if true, uses true lfdrs as a benchmark
DO_COMPETITORS = True  # If true, runs with the competitors from the paper
ONLY_SPATIAL = False  # If true, plots only results for spatial varying prior

# Number of workers for the parallelization
num_wrk = np.min((50, os.cpu_count() - 1))  # Change first value if you want to
# use more than 50 cores if available.

PLOT_EXAMPLE_RUNS = True
PLOT_PERFORMANCE_METRICS = True

# %% setup: adjusting illegal parameter choices
# there are no ground true lfdrs available for custom p-values
if not FIELD_MODE == 'radio':
    DO_TRUE = False
if FIELD_MODE == 'non-spatial':
    DO_SPATIAL_PARTITION = False
    DO_VAR_SPATIAL_PRIOR = False
    ONLY_SPATIAL = False
# %% setup: hypothesis testing parameters
# Specify here all nominal FDR levels that results shall be computed for!
alp_vec = np.array([0.01, 0.02, 0.05, 0.07, 0.10, 0.15, 0.2, .25, .3])

# For example plots: which MC run shall be shown?
mc = 0

# %% setup: create directories to store the results
if not SAVE_RESULTS:
    print("Warning! The results will not be saved.")
else:
    # create directories to store the results, if they don't exist
    dat_path = paths.get_path_to_dat(FD_SCEN)
    res_path = paths.get_path_to_res(FD_SCEN)
    try:
        os.makedirs(dat_path)
    except FileExistsError:
        None
    try:
        if not (SEN_CFG == 'custom' or SEN_CFG == 'non-spatial'):
            os.makedirs(os.path.join(dat_path, SEN_CFG))
    except FileExistsError:
        None
    try:
        os.makedirs(res_path)
    except FileExistsError:
        None
    try:
        os.makedirs(os.path.join(res_path, SEN_CFG))
    except FileExistsError:
        None
    if QUANTIZE:
        quan_path = os.path.join(
            res_path, SEN_CFG, f'quan_{N_QUAN_BITS}Bit',
            f'sensoring_at_{SENSORING_LAM}')
        try:
            os.makedirs(quan_path)
        except FileExistsError:
            None

# %% setup: read in parameters
# Read in field parameters
fd_dim, n_MC, n_sam, n_src, pi0_des, sha_fa, prop_env = par.get_par_fd_scen(
    FD_SCEN, dat_path)

# Read in sensor parameters
if FIELD_MODE == 'custom' or FIELD_MODE == 'non-spatial':
    n_sen, n_sam_per_sen, var_sen_loc, sen_hom = par.get_par_sen_cfg(
        FD_SCEN, dat_path)
else:
    n_sen, n_sam_per_sen, var_sen_loc, sen_hom = par.get_par_sen_cfg(
        SEN_CFG, dat_path)

if np.max(n_sam_per_sen) > n_sam:
    # We cannot use more samples per sensor than there are samples for each
    # grid point.
    print("Change the sensor configuration! Cannot use more samples per sensor"
          " than there are samples per grid point")
    sys.exit()

# Read in parameters for the spectral method of moments
sMoM_k, sMoM_d, sMoM_n_tr, sMoM_reps_eta, sMoM_dis_msr = par.get_par_smom(
    dat_path)

# Read in parameters for the spatial partitioning.
(spa_var_bw_grid, spa_var_ker_vec, spa_var_sthr_grid,
 spa_var_pi0_max) = par.get_par_spa_var(dat_path)

print(f'Running scenario {FD_SCEN} in sensor cfg {SEN_CFG}!')
# %% setup: reading in or creating the data
if 'fd' in globals() and 'est_fd' in globals():
    print('Fields already loaded')
else:
    fd, est_fd = fd_hdl.rd_in_fds(FD_SCEN, SEN_CFG, dat_path)

fully_loaded = fd.n == est_fd.n
if fully_loaded:
    print('In this configuration, a sensor is placed at each grid point!',
          'No interpolation necessary.')
else:
    DO_SPATIAL_PARTITION = False
# %% the true lfdrs at all grid points
# For reference, does not require any interpolation.
# Only implemented for p-values generated from our data generator. If you
# provide own p-values, you must provide own true lfdrs (if they are available)
# and replace this part.
if DO_TRUE:
    # load/compute
    [lfdrs, f_p, f1_p_sen, pi0] = lfdr_est.get_true_lfdrs(
        fd, os.path.join(res_path), SAVE_RESULTS)

    # create detection results
    det_res_tru = det.apply_lfdr_detection(lfdrs, fd.r_tru, alp_vec,
                                           'true at grid points', sen=True)

# %% quantization
if QUANTIZE:
    (q_borders, q_width, q_centers) = par.get_par_quan(N_QUAN_BITS, dat_path,
                                                       lam=SENSORING_LAM)
    fd.quantize_p_and_z(q_borders, q_centers)
    est_fd.quantize_p_and_z(q_borders, q_centers)
    res_path = quan_path
else:
    res_path = os.path.join(res_path, SEN_CFG)

# %% the true lfdrs at the sensors
# To evaluate the quality of the resulting lfdr estimates.
# Only implemented for p-values generated from our data generator. If you
# provide own p-values, you must provide own true lfdrs (if they are available)
# and replace this part.
if DO_TRUE:
    # load/compute
    [lfdrs_sen, f_p_sen, f1_p_sen, pi0_sen] = lfdr_est.get_true_lfdrs(
        est_fd, res_path, SAVE_RESULTS)

    # interpolate
    if fully_loaded:
        lfdrs_ipl = lfdrs_sen
    else:
        lfdrs_ipl = lfdr_est.ipl_lfdrs(res_path, 'ground-truth-sen', lfdrs_sen,
                                       est_fd.sen_cds, est_fd.dim, fd.n)

    # create detection results
    det_res_sen_tru = det.apply_lfdr_detection(
        lfdrs_sen, est_fd.r_tru, alp_vec, 'true at sensors', sen=True)
    det_res_ipl_tru = det.apply_lfdr_detection(
        lfdrs_ipl, fd.r_tru, alp_vec, 'true interpolated', sen=False)

# %% competitors at sensors
if DO_COMPETITORS:
    # Get sensor lfdrs
    [lfdrs_sen_lm, f_p_sen_lm, f1_p_sen_lm, pi0_sen_lm, ex_time_sen_lm] = (
        lfdr_est.est_lfdrs(est_fd, res_path, SAVE_RESULTS, "lm", 0.5))
    [lfdrs_sen_gmm, f_p_sen_gmm, f1_p_sen_gmm, pi0_sen_gmm,
     ex_time_sen_gmm, num_sel_cmp_sen_gmm] = lfdr_est.est_lfdrs(
        est_fd, res_path, SAVE_RESULTS, "gmm", 0.5)
    [lfdrs_sen_pr, f_p_sen_pr, f1_p_sen_pr, pi0_sen_pr, ex_time_sen_pr] = (
        lfdr_est.est_lfdrs(est_fd, res_path, SAVE_RESULTS, "pr", [10, -50]))
    [lfdrs_sen_bum, f_p_sen_bum, f1_p_sen_bum, pi0_sen_bum,
     ex_time_sen_bum] = (lfdr_est.est_lfdrs(
         est_fd, res_path, SAVE_RESULTS, "bum",
         [np.arange(0.001, 0.3, 0.01), np.arange(0.6, 1.01, 0.01)]))

    # Detection results at sensors only.
    det_res_sen_lm = det.apply_lfdr_detection(
        lfdrs_sen_lm, est_fd.r_tru, alp_vec, 'LM at sensors', sen=True)
    det_res_sen_gmm = det.apply_lfdr_detection(
        lfdrs_sen_gmm, est_fd.r_tru, alp_vec, 'GMM at sensors', sen=True)
    det_res_sen_pr = det.apply_lfdr_detection(
        lfdrs_sen_pr, est_fd.r_tru, alp_vec, 'PR at sensors', sen=True)
    det_res_sen_bum = det.apply_lfdr_detection(
        lfdrs_sen_bum, est_fd.r_tru, alp_vec, 'BUM at sensors', sen=True)
    det_res_sen_dbh = det.apply_dBH(est_fd.p, est_fd.r_tru, alp_vec)

    if fully_loaded:
        lfdrs_ipl_lm = lfdrs_sen_lm
        lfdrs_ipl_gmm = lfdrs_sen_gmm
        lfdrs_ipl_pr = lfdrs_sen_pr
        lfdrs_ipl_bum = lfdrs_sen_bum
    else:
        # Get lfdrs at all grid points by TPS interpolation
        lfdrs_ipl_lm = lfdr_est.ipl_lfdrs(
            res_path, 'lm-sen', lfdrs_sen_lm, est_fd.sen_cds, est_fd.dim, fd.n)
        lfdrs_ipl_gmm = lfdr_est.ipl_lfdrs(
            res_path, 'gmm-sen', lfdrs_sen_gmm, est_fd.sen_cds, est_fd.dim,
            fd.n)
        lfdrs_ipl_pr = lfdr_est.ipl_lfdrs(
            res_path, 'pr-sen', lfdrs_sen_pr, est_fd.sen_cds, est_fd.dim, fd.n)
        lfdrs_ipl_bum = lfdr_est.ipl_lfdrs(
            res_path, 'bum-sen', lfdrs_sen_bum, est_fd.sen_cds, est_fd.dim,
            fd.n)

    # Detection results at all grid points
    det_res_ipl_lm = det.apply_lfdr_detection(
        lfdrs_ipl_lm, fd.r_tru, alp_vec, 'LM at all grid points', sen=False)
    det_res_ipl_gmm = det.apply_lfdr_detection(
        lfdrs_ipl_gmm, fd.r_tru, alp_vec, 'GMM at all grid points', sen=False)
    det_res_ipl_pr = det.apply_lfdr_detection(
        lfdrs_ipl_pr, fd.r_tru, alp_vec, 'PR at all grid points', sen=False)
    det_res_ipl_bum = det.apply_lfdr_detection(
        lfdrs_ipl_bum, fd.r_tru, alp_vec, 'BUM at all grid points', sen=False)

# %% application of lfdr-sMoM
if DO_SPATIAL_PARTITION:
    [lfdrs_sen_smom_s, f_p_sen_smom_s, f1_p_sen_smom_s, pi0_sen_smom_s,
     ex_time_sen_smom_s] = (lfdr_est.est_lfdrs(
        est_fd, res_path, SAVE_RESULTS, "smom_s",
        [dat_path, 50, 'stan', N_QUAN_BITS, SENSORING_LAM]))

    det_res_sen_smom_s = det.apply_lfdr_detection(
        lfdrs_sen_smom_s, est_fd.r_tru, alp_vec, 'lfdr-sMoM_s at sensors',
        sen=True)

    if fully_loaded:
        lfdrs_ipl_smom_s = lfdrs_sen_smom_s
    else:
        # Get lfdrs at all grid points by TPS interpolation
        lfdrs_ipl_smom_s = lfdr_est.ipl_lfdrs(
            res_path, 'smom_s-sen', lfdrs_sen_smom_s, est_fd.sen_cds,
            est_fd.dim, fd.n)

    # Detection results at all grid points
    det_res_ipl_smom_s = det.apply_lfdr_detection(
        lfdrs_ipl_smom_s, fd.r_tru, alp_vec, 'lfdr-sMoM_s at all grid points',
        sen=False)

[lfdrs_sen_smom, f_p_sen_smom, f1_p_sen_smom, pi0_sen_smom,
 ex_time_sen_smom] = (lfdr_est.est_lfdrs(
     est_fd, res_path, SAVE_RESULTS, "smom",
     [dat_path, 50, 'stan', N_QUAN_BITS, SENSORING_LAM]))
det_res_sen_smom = det.apply_lfdr_detection(
    lfdrs_sen_smom, est_fd.r_tru, alp_vec, 'lfdr-sMoM at sensors',
    sen=True)

if fully_loaded:
    lfdrs_ipl_smom = lfdrs_sen_smom
else:
    # Get lfdrs at all grid points by TPS interpolation
    lfdrs_ipl_smom = lfdr_est.ipl_lfdrs(
        res_path, 'smom-sen', lfdrs_sen_smom, est_fd.sen_cds, est_fd.dim, fd.n)

# Detection results at all grid points
det_res_ipl_smom = det.apply_lfdr_detection(
    lfdrs_ipl_smom, fd.r_tru, alp_vec, 'lfdr-sMoM at all grid points',
    sen=False)

# %% application of lfdr-MBM-EM
if DO_COMPETITORS and DO_EM:
    mbm_par_type = 'stan'
    # Do lfdr-MBM-EM
    mbm_em_mod_ords, mbm_em_reps, mbm_em_cvg_thr = par.get_par_mbm_em(dat_path)

    [lfdrs_sen_mbm_em, f_p_sen_mbm_em, f1_p_sen_mbm_em, pi0_sen_mbm_em,
     ex_time_sen_mbm_em] = (lfdr_est.est_lfdrs(
         est_fd, res_path, SAVE_RESULTS, "mbm-em",
         [dat_path, 50, mbm_par_type]))
    det_res_sen_mbm_em = det.apply_lfdr_detection(
        lfdrs_sen_mbm_em, est_fd.r_tru, alp_vec, 'lfdr-MBM-EM at sensors',
        sen=True)

    if fully_loaded:
        lfdrs_ipl_mbm_em = lfdrs_sen_mbm_em
    else:
        # Get lfdrs at all grid points by TPS interpolation
        lfdrs_ipl_mbm_em = lfdr_est.ipl_lfdrs(
            res_path, mbm_par_type + '_mbm-em-sen', lfdrs_sen_mbm_em,
            est_fd.sen_cds, est_fd.dim, fd.n)

    # Detection results at all grid points
    det_res_ipl_mbm_em = det.apply_lfdr_detection(
        lfdrs_ipl_mbm_em, fd.r_tru, alp_vec, 'lfdr-MBM-EM at all grid points',
        sen=False)

# %% application of lfdr-sMoM-EM
if DO_SPATIAL_PARTITION and DO_EM:
    if fd.n == est_fd.n:
        [lfdrs_sen_smom_s_em, f_p_sen_smom_s_em, f1_p_sen_smom_s_em,
         pi0_sen_smom_s_em, ex_time_sen_smom_s_em] = (lfdr_est.est_lfdrs(
            est_fd, res_path, SAVE_RESULTS, "smom_s-em",
            [50, 1e-5]))
    else:
        print(['Spatial partitioning only implemented for cfgs where a sensor '
               'is located at each grid point. '
               'Set DO_SPATIAL_PARTITION=False!'])
        sys.exit()

    det_res_sen_smom_s_em = det.apply_lfdr_detection(
        lfdrs_sen_smom_s_em, est_fd.r_tru, alp_vec,
        'lfdr-sMoM_s-EM at sensors', sen=True)

    if fully_loaded:
        lfdrs_ipl_smom_s_em = lfdrs_sen_smom_s_em
    else:
        # Get lfdrs at all grid points by TPS interpolation
        lfdrs_ipl_smom_s_em = lfdr_est.ipl_lfdrs(
            res_path, 'smom_s-em-sen', lfdrs_sen_smom_s_em, est_fd.sen_cds,
            est_fd.dim, fd.n)

    # Detection results at all grid points
    det_res_ipl_smom_s_em = det.apply_lfdr_detection(
        lfdrs_ipl_smom_s_em, fd.r_tru, alp_vec,
        'lfdr-sMoM_s-EM at all grid points', sen=False)

if DO_EM:
    [lfdrs_sen_smom_em, f_p_sen_smom_em, f1_p_sen_smom_em, pi0_sen_smom_em,
     ex_time_sen_smom_em] = (lfdr_est.est_lfdrs(
         est_fd, res_path, SAVE_RESULTS, "smom-em",
         [50, 1e-5]))
    det_res_sen_smom_em = det.apply_lfdr_detection(
        lfdrs_sen_smom_em, est_fd.r_tru, alp_vec, 'lfdr-sMoM-EM at sensors',
        sen=True)

    if fully_loaded:
        lfdrs_ipl_smom_em = lfdrs_sen_smom_em
    else:
        # Get lfdrs at all grid points by TPS interpolation
        lfdrs_ipl_smom_em = lfdr_est.ipl_lfdrs(
            res_path, 'smom-em-sen', lfdrs_sen_smom_em, est_fd.sen_cds,
            est_fd.dim, fd.n)

    # Detection results at all grid points
    det_res_ipl_smom_em = det.apply_lfdr_detection(
        lfdrs_ipl_smom_em, fd.r_tru, alp_vec,
        'lfdr-sMoM-EM at all grid points', sen=False)

# %% application of clfdr-sMoM-SLS and clfdr-sMoM-SNS
if DO_VAR_SPATIAL_PRIOR:
    [clfdrs_sen_smom_sls, pi0_sen_smom_sls] = (lfdr_est.est_clfdrs(
        est_fd, res_path, SAVE_RESULTS, "smom-sls",
        [dat_path, 50, 'stan', 'smom-sen']))
    det_res_sen_smom_sls = det.apply_lfdr_detection(
        clfdrs_sen_smom_sls, est_fd.r_tru, alp_vec,
        'clfdr-sMoM-SLS at sensors', sen=True)
    [clfdrs_sen_smom_sns, pi0_sen_smom_sns] = (lfdr_est.est_clfdrs(
        est_fd, res_path, SAVE_RESULTS, "smom-sns",
        [dat_path, 50, 'stan', 'smom-sen']))
    det_res_sen_smom_sns = det.apply_lfdr_detection(
        clfdrs_sen_smom_sns, est_fd.r_tru, alp_vec,
        'clfdr-sMoM-SNS at sensors', sen=True)

    if fully_loaded:
        clfdrs_ipl_smom_sls = clfdrs_sen_smom_sls
        clfdrs_ipl_smom_sns = clfdrs_sen_smom_sns
    else:
        # Get lfdrs at all grid points by TPS interpolation
        clfdrs_ipl_smom_sls = lfdr_est.ipl_lfdrs(
            res_path, 'smom-sen-sls', clfdrs_sen_smom_sls, est_fd.sen_cds,
            est_fd.dim, fd.n)
        clfdrs_ipl_smom_sns = lfdr_est.ipl_lfdrs(
            res_path, 'smom-sen-sns', clfdrs_sen_smom_sns, est_fd.sen_cds,
            est_fd.dim, fd.n)

    # Detection results at all grid points
    det_res_ipl_smom_sls = det.apply_lfdr_detection(
        clfdrs_ipl_smom_sls, fd.r_tru, alp_vec,
        'clfdr-sMoM-SLS at all grid points', sen=False)
    det_res_ipl_smom_sns = det.apply_lfdr_detection(
        clfdrs_ipl_smom_sns, fd.r_tru, alp_vec,
        'clfdr-sMoM-SNS at all grid points', sen=False)

# %% application of FDRS
# if DO_VAR_SPATIAL_PRIOR and DO_COMPETITORS:
    # [clfdrs_sen_fdrs, pi0_sen_fdrs] = (lfdr_est.est_clfdrs(
    #     est_fd, res_path, SAVE_RESULTS, "fdrs", 50))
    # det_res_sen_fdrs = det.apply_lfdr_detection(
    #     clfdrs_sen_fdrs, est_fd.r_tru, alp_vec,
    #     'FDRS at sensors', sen=True)

    # if fully_loaded:
    #     clfdrs_ipl_fdrs = clfdrs_sen_fdrs
    # else:
    #     # Get lfdrs at all grid points by TPS interpolation
    #     clfdrs_ipl_fdrs = lfdr_est.ipl_lfdrs(
    #         res_path, 'fdrs', clfdrs_sen_fdrs, est_fd.sen_cds, est_fd.dim,
    #         fd.n)

    # # Detection results at all grid points
    # det_res_ipl_fdrs = det.apply_lfdr_detection(
    #     clfdrs_ipl_fdrs, fd.r_tru, alp_vec,
    #     'FDRS', sen=False)
# %% application of clfdr-sMoM-EM-SLS and clfdr-sMoM-EM-SNS
if DO_VAR_SPATIAL_PRIOR and DO_EM:
    [clfdrs_sen_smom_em_sls, pi0_sen_smom_em_sls] = (lfdr_est.est_clfdrs(
        est_fd, res_path, SAVE_RESULTS, "smom-sls",
        [dat_path, 50, 'stan', 'smom-em-sen']))
    det_res_sen_smom_em_sls = det.apply_lfdr_detection(
        clfdrs_sen_smom_em_sls, est_fd.r_tru, alp_vec,
        'clfdr-sMoM-EM-SLS at sensors', sen=True)
    [clfdrs_sen_smom_em_sns, pi0_sen_smom_em_sns] = (lfdr_est.est_clfdrs(
        est_fd, res_path, SAVE_RESULTS, "smom-sns",
        [dat_path, 50, 'stan', 'smom-em-sen']))
    det_res_sen_smom_em_sns = det.apply_lfdr_detection(
        clfdrs_sen_smom_em_sns, est_fd.r_tru, alp_vec,
        'clfdr-sMoM-EM-SNS at sensors', sen=True)

    if fully_loaded:
        clfdrs_ipl_smom_em_sls = clfdrs_sen_smom_em_sls
        clfdrs_ipl_smom_em_sns = clfdrs_sen_smom_em_sns
    else:
        # Get lfdrs at all grid points by TPS interpolation
        clfdrs_ipl_smom_em_sls = lfdr_est.ipl_lfdrs(
            res_path, 'smom-em-sen-sls', clfdrs_sen_smom_em_sls,
            est_fd.sen_cds, est_fd.dim, fd.n)
        clfdrs_ipl_smom_em_sns = lfdr_est.ipl_lfdrs(
            res_path, 'smom-em-sen-sns', clfdrs_sen_smom_em_sns,
            est_fd.sen_cds, est_fd.dim, fd.n)
    # Detection results at all grid points
    det_res_ipl_smom_em_sls = det.apply_lfdr_detection(
        clfdrs_ipl_smom_em_sls, fd.r_tru, alp_vec,
        'clfdr-sMoM-EM-SLS at all grid points', sen=False)
    det_res_ipl_smom_em_sns = det.apply_lfdr_detection(
        clfdrs_ipl_smom_em_sns, fd.r_tru, alp_vec,
        'clfdr-sMoM-EM-SNS at all grid points', sen=False)

# %% Check if ground truth is available and hence, performance measures were
# computable
if np.sum(np.isnan(est_fd.r_tru)) > 0:
    PLOT_PERFORMANCE_METRICS = False

# %% plots: Fig. 4 from [Goelz2022TSIPN]/ Fig. 2 from [Goelz2022ICASSP]:
# Exemplary detection results + lfdrs at all grid points with interpolation
# REMARK: The settings for the figure in [Goelz2022TSIPN] were:
# FD_SCEN = scB_TSIPN, SEN_CFG = cfg2_TSIPN, show_lfdrs = lfdrs_ipl_smom,
# show_res = det_res_ipl_smom, show_alp_val = 0.1
# REMARK: The settings for the figure in [Goelz2022ICASSP] were:
# FD_SCEN = scB_ICASSP, SEN_CFG = stan_300, show_lfdrs = lfdrs_ipl_smom,
# show_res = det_res_ipl_smom, show_alp_val = 0.1
if PLOT_EXAMPLE_RUNS:
    try:
        show_alp_val = .1
        show_lfdrs = lfdrs_ipl_smom
        show_res = det_res_ipl_smom
        anal.plot_lfdrs(show_lfdrs, fd, mc=mc)
        plt.title('Fig. 4b [Goelz2022TISPN] or Fig. 2b [Goelz2022ICASSP] with '
                  f'{FD_SCEN} in cfg {SEN_CFG} and '
                  f'method {{show_res[0].nam}}')

        show_alp_vec_idx = np.where(alp_vec == show_alp_val)[0][0]
        anal.plot_rej(show_res[show_alp_vec_idx], fd, n_MC=mc)
        plt.title('Fig. 4a [Goelz2022TISPN] or Fig. 2b [Goelz2022ICASSP] with '
                  f'{FD_SCEN} in cfg {SEN_CFG} and '
                  f'method {show_res[show_alp_vec_idx].nam}')
        del show_alp_val, show_lfdrs, show_res, show_alp_vec_idx
    except NameError:
        print('Results for method chosen as example plot do not exist!')

# %% plots: Similar to Fig. 4 from [Goelz2022CISS]: Exemplary detection
# results + lfdrs but only at sensors (Remark: call function like this
# (with est_fd))
# REMARK: The settings for the figure in the paper were: FD_SCEN = scA_CISS,
# SEN_CFG = stan_300,
# show_lfdrs = one of {lfdrs_ipl_smom, clfdrs_ipl_fdrs, clfdrs_ipl_smom_sns,
# clfdrs_ipl_smom_sls},
# show_res = one of {det_res_ipl_smom, det_res_ipl_fdrs, det_res_ipl_smom_sns,
# det_res_ipl_smom_sls},
# show_alp_val = 0.1
if PLOT_EXAMPLE_RUNS:
    try:
        show_alp_val = .1
        show_lfdrs = clfdrs_sen_smom_sns
        show_res = det_res_sen_smom_sns
        anal.plot_lfdrs(show_lfdrs, est_fd, mc=mc)
        plt.title(f'Fig. 4b [Goelz2022CISS] with {FD_SCEN} in cfg {SEN_CFG} '
                  f'and method {show_res[0].nam}')

        show_alp_vec_idx = np.where(alp_vec == show_alp_val)[0][0]
        anal.plot_rej(show_res[show_alp_vec_idx], est_fd, n_MC=mc)
        plt.title(f'Fig. 4a [Goelz2022CISS] with {FD_SCEN} in cfg {SEN_CFG} '
                  f'and method {show_res[show_alp_vec_idx].nam}')
        del show_alp_val, show_lfdrs, show_res, show_alp_vec_idx
    except NameError:
        print('Results for method chosen as example plot do not exist!')



# %% plots: Fig. 5 from [Goelz2022TSIPN], Fig. 5 from [Goelz2022CISS] and
# Fig 3 from [Goelz2022ICASSP]: FDR and power over nominal levels
# to make the plot look like in the paper, only at sensors!
# REMARK: The settings for Fig. 5  from [Goelz2022TSIPN] were:
# FD_SCEN = one of {scA_TSIPN, scB_TSIPN, scC_TSIPN},
# SEN_CFG = cfg1_TSIPN, DO_EM = False, DO_COMPETITORS = True, DO_TRUE = True,
# DO_SPATIAL_PARTITION = True, DO_VAR_SPATIAL_PRIOR = False, ONLY_SPATIAL=False
# REMARK: The settings for Fig. 5  from [Goelz2022CISS] were:
# FD_SCEN = one of {scA_CISS, scB_CISS, scC_CISS},
# SEN_CFG = stan_3000, DO_EM = True, DO_COMPETITORS = True, DO_TRUE = True,
# DO_SPATIAL_PARTITION = False, DO_VAR_SPATIAL_PRIOR = False,
# ONLY_SPATIAL=False
# REMARK: The settings for Fig. 3  from [Goelz2022ICASSP] were:
# FD_SCEN = scC_ICASSP,
# SEN_CFG = stan_1000, DO_EM = False, DO_COMPETITORS = True, DO_TRUE = False,
# DO_SPATIAL_PARTITION = False, DO_VAR_SPATIAL_PRIOR = True, ONLY_SPATIAL=True
linestyle_lst = []
color_lst = []
plot_label_lst = []
plotting_lst = []
if DO_TRUE:
    linestyle_lst.append('s--')
    color_lst.append('black')
    plot_label_lst.append('True lfdrs')
    plotting_lst.append(det_res_sen_tru)
if DO_COMPETITORS and not ONLY_SPATIAL:
    linestyle_lst.insert(0, '+--')
    color_lst.insert(0, 'tab:gray')
    plot_label_lst.insert(0, 'dBH')
    plotting_lst.insert(0, det_res_sen_dbh)
    linestyle_lst.extend(['-.', '-.', '-.', '-.'])
    color_lst.extend(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    plot_label_lst.extend(['LM', 'BUM', 'PR', 'GMM'])
    plotting_lst.extend([det_res_sen_lm, det_res_sen_bum, det_res_sen_pr,
                         det_res_sen_gmm])
if DO_SPATIAL_PARTITION:
    linestyle_lst.append('^--')
    color_lst.append('tab:brown')
    plot_label_lst.append('lfdr-sMoM_s')
    plotting_lst.append(det_res_sen_smom_s)

linestyle_lst.append('v--')
color_lst.append('tab:pink')
plot_label_lst.append('lfdr-sMoM')
plotting_lst.append(det_res_sen_smom)

if DO_COMPETITORS and DO_EM and not ONLY_SPATIAL:
    linestyle_lst.append('d--')
    color_lst.append('tab:gray')
    plot_label_lst.append('lfdr-MBM-EM')
    plotting_lst.append(det_res_sen_mbm_em)

if DO_SPATIAL_PARTITION and DO_EM:
    linestyle_lst.append('*--')
    color_lst.append('tab:brown')
    plot_label_lst.append('lfdr-sMoM_s-EM')
    plotting_lst.append(det_res_sen_smom_s_em)

if DO_EM:
    linestyle_lst.append('*--')
    color_lst.append('tab:pink')
    plot_label_lst.append('lfdr-sMoM-EM')
    plotting_lst.append(det_res_sen_smom_em)

# if DO_VAR_SPATIAL_PRIOR and DO_COMPETITORS:
#     linestyle_lst.append('-.')
#     color_lst.append('purple')
#     plot_label_lst.append('FDRS')
#     plotting_lst.append(det_res_sen_fdrs)

if DO_VAR_SPATIAL_PRIOR:
    linestyle_lst.append('o--')
    color_lst.append('green')
    plot_label_lst.append('clfdr-sMoM-SLS')
    plotting_lst.append(det_res_sen_smom_sls)

    linestyle_lst.append('o--')
    color_lst.append('red')
    plot_label_lst.append('clfdr-sMoM-SNS')
    plotting_lst.append(det_res_sen_smom_sns)

if PLOT_PERFORMANCE_METRICS:
    anal.plot_fdrs_and_pow(alp_vec, plotting_lst, linestyle_lst, color_lst,
                           plot_label_lst)

# %% plots: Similar to Fig. 5 from [Goelz2022TSIPN]: FDR and power over nominal
# levels, but at all grid points. Values from this plot are displayed in
# Table I and Table II of [Goelz2022TSIPN].
linestyle_lst = []
color_lst = []
plot_label_lst = []
plotting_lst = []
if DO_TRUE:
    linestyle_lst.append('s--')
    color_lst.append('black')
    plot_label_lst.append('True lfdrs')
    plotting_lst.append(det_res_ipl_tru)
if DO_COMPETITORS and not ONLY_SPATIAL:
    linestyle_lst.extend(['-.', '-.', '-.', '-.'])
    color_lst.extend(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    plot_label_lst.extend(['LM', 'BUM', 'PR', 'GMM'])
    plotting_lst.extend([det_res_ipl_lm, det_res_ipl_bum, det_res_ipl_pr,
                         det_res_ipl_gmm])
if DO_SPATIAL_PARTITION:
    linestyle_lst.append('^--')
    color_lst.append('tab:brown')
    plot_label_lst.append('lfdr-sMoM_s')
    plotting_lst.append(det_res_ipl_smom_s)

linestyle_lst.append('v--')
color_lst.append('tab:pink')
plot_label_lst.append('lfdr-sMoM')
plotting_lst.append(det_res_ipl_smom)

if DO_COMPETITORS and DO_EM and not ONLY_SPATIAL:
    linestyle_lst.append('d--')
    color_lst.append('tab:gray')
    plot_label_lst.append('lfdr-MBM-EM')
    plotting_lst.append(det_res_ipl_mbm_em)

if DO_SPATIAL_PARTITION and DO_EM:
    linestyle_lst.append('*--')
    color_lst.append('tab:brown')
    plot_label_lst.append('lfdr-sMoM_s-EM')
    plotting_lst.append(det_res_ipl_smom_s_em)

if DO_EM:
    linestyle_lst.append('*--')
    color_lst.append('tab:pink')
    plot_label_lst.append('lfdr-sMoM-EM')
    plotting_lst.append(det_res_ipl_smom_em)

# if DO_VAR_SPATIAL_PRIOR and DO_COMPETITORS:
#     linestyle_lst.append('-.')
#     color_lst.append('purple')
#     plot_label_lst.append('FDRS')
#     plotting_lst.append(det_res_ipl_fdrs)

if DO_VAR_SPATIAL_PRIOR:
    linestyle_lst.append('o--')
    color_lst.append('green')
    plot_label_lst.append('clfdr-sMoM-SLS')
    plotting_lst.append(det_res_ipl_smom_sls)

    linestyle_lst.append('o--')
    color_lst.append('red')
    plot_label_lst.append('clfdr-sMoM-SNS')
    plotting_lst.append(det_res_ipl_smom_sns)

if PLOT_PERFORMANCE_METRICS:
    anal.plot_fdrs_and_pow(alp_vec, plotting_lst, linestyle_lst, color_lst,
                           plot_label_lst)

# %% analysis: print execution times (allows comparison of methods as in Fig. 5
# of [Goelz2022TSIPN])
# Note: Execution times are only available for lfdr-based procedures, since the
# lfdr estimation process is what we time.
print('')
ex_times_lst = []
times_plot_label_lst = []
if DO_COMPETITORS:
    times_plot_label_lst.extend(['LM', 'GMM', 'BUM', 'PR'])
    ex_times_lst.extend([ex_time_sen_lm, ex_time_sen_gmm, ex_time_sen_bum,
                         ex_time_sen_pr])
if DO_SPATIAL_PARTITION:
    ex_times_lst.append(ex_time_sen_smom_s)
    times_plot_label_lst.append('lfdr-sMoM_s')

if DO_COMPETITORS and DO_EM:
    times_plot_label_lst.append('lfdr-MBM-EM')
    ex_times_lst.append(ex_time_sen_mbm_em)

if DO_SPATIAL_PARTITION and DO_EM:
    times_plot_label_lst.append('lfdr-sMoM_s-EM')
    ex_times_lst.append(ex_time_sen_smom_s_em)

ex_times_lst.append(ex_time_sen_smom)
times_plot_label_lst.append('lfdr-sMoM')

if DO_EM:
    times_plot_label_lst.append('lfdr-sMoM-EM')
    ex_times_lst.append(ex_time_sen_smom_em)


anal.print_ex_times(ex_times_lst, times_plot_label_lst)
print('')

# %% analysis: print FDR and detection power for different applied methods as
# in Tab. 2 of [Goelz2022ICASSP] for interpolated fields!
# REMARK: The settings for Tab. 2 of [Goelz2022ICASSP] were:
# FD_SCEN = one of {scA_ICASSP, scB_ICASSP, scC_ICASSP},
# SEN_CFG = one of {stan_300, stan_1000, stan_3000}, DO_EM = False,
# DO_COMPETITORS = True, DO_TRUE = False,
# DO_SPATIAL_PARTITION = False, DO_VAR_SPATIAL_PRIOR = Trye, ONLY_SPATIAL=True,
# show_alp_val = .1
show_alp_val = .1
alp_vec_idx = np.where(alp_vec == show_alp_val)[0][0]
table_res_list = []
table_label_lst = []
if DO_TRUE:
    table_label_lst.append('True lfdrs')
    table_res_list.append(det_res_ipl_tru[alp_vec_idx])
if DO_COMPETITORS and not ONLY_SPATIAL:
    table_label_lst.extend(['LM', 'BUM', 'PR', 'GMM'])
    table_res_list.extend([det_res_ipl_lm[alp_vec_idx],
                           det_res_ipl_bum[alp_vec_idx],
                           det_res_ipl_pr[alp_vec_idx],
                           det_res_ipl_gmm[alp_vec_idx]])
if DO_SPATIAL_PARTITION:
    table_label_lst.append('lfdr-sMoM_s')
    table_res_list.append(det_res_ipl_smom_s[alp_vec_idx])

table_label_lst.append('lfdr-sMoM')
table_res_list.append(det_res_ipl_smom[alp_vec_idx])

if DO_COMPETITORS and DO_EM and not ONLY_SPATIAL:
    table_label_lst.append('lfdr-MBM-EM')
    table_res_list.append(det_res_ipl_mbm_em[alp_vec_idx])

if DO_SPATIAL_PARTITION and DO_EM:
    table_label_lst.append('lfdr-sMoM_s-EM')
    table_res_list.append(det_res_ipl_smom_s_em[alp_vec_idx])

if DO_EM:
    table_label_lst.append('lfdr-sMoM-EM')
    table_res_list.append(det_res_ipl_smom_em[alp_vec_idx])

# if DO_VAR_SPATIAL_PRIOR and DO_COMPETITORS:
#     table_label_lst.append('FDRS')
#     table_res_list.append(det_res_ipl_fdrs[alp_vec_idx])

if DO_VAR_SPATIAL_PRIOR:
    table_label_lst.append('clfdr-sMoM-SLS')
    table_res_list.append(det_res_ipl_smom_sls[alp_vec_idx])

    table_label_lst.append('clfdr-sMoM-SNS')
    table_res_list.append(det_res_ipl_smom_sns[alp_vec_idx])

if PLOT_PERFORMANCE_METRICS:
    print("PERFORMANCE MEASURES WITH MHT AT ALL GRID POINTS")
    anal.print_fdrs_and_pow(alp_vec[alp_vec_idx], table_res_list,
                            table_label_lst)
print('')
# %% analysis: print FDR and detection power for different applied methods
# similar to Tab. 2 of [Goelz2022ICASSP], but only at the sensors!
show_alp_val = .1
alp_vec_idx = np.where(alp_vec == show_alp_val)[0][0]
table_res_list = []
table_label_lst = []
if DO_TRUE:
    table_label_lst.append('True lfdrs')
    table_res_list.append(det_res_sen_tru[alp_vec_idx])
if DO_COMPETITORS and not ONLY_SPATIAL:
    table_label_lst.extend(['LM', 'BUM', 'PR', 'GMM'])
    table_res_list.extend([det_res_sen_lm[alp_vec_idx],
                           det_res_sen_bum[alp_vec_idx],
                           det_res_sen_pr[alp_vec_idx],
                           det_res_sen_gmm[alp_vec_idx]])
if DO_SPATIAL_PARTITION:
    table_label_lst.append('lfdr-sMoM_s')
    table_res_list.append(det_res_sen_smom_s[alp_vec_idx])

table_label_lst.append('lfdr-sMoM')
table_res_list.append(det_res_sen_smom[alp_vec_idx])

if DO_COMPETITORS and DO_EM and not ONLY_SPATIAL:
    table_label_lst.append('lfdr-MBM-EM')
    table_res_list.append(det_res_sen_mbm_em[alp_vec_idx])

if DO_SPATIAL_PARTITION and DO_EM:
    table_label_lst.append('lfdr-sMoM_s-EM')
    table_res_list.append(det_res_sen_smom_s_em[alp_vec_idx])

if DO_EM:
    table_label_lst.append('lfdr-sMoM-EM')
    table_res_list.append(det_res_sen_smom_em[alp_vec_idx])

# if DO_VAR_SPATIAL_PRIOR and DO_COMPETITORS:
#     table_label_lst.append('FDRS')
#     table_res_list.append(det_res_sen_fdrs[alp_vec_idx])

if DO_VAR_SPATIAL_PRIOR:
    table_label_lst.append('clfdr-sMoM-SLS')
    table_res_list.append(det_res_ipl_smom_sls[alp_vec_idx])

    table_label_lst.append('clfdr-sMoM-SNS')
    table_res_list.append(det_res_ipl_smom_sns[alp_vec_idx])

if PLOT_PERFORMANCE_METRICS:
    print("PERFORMANCE MEASURES WITH MHT AT THE SENSORS")
    anal.print_fdrs_and_pow(alp_vec[alp_vec_idx], table_res_list,
                            table_label_lst)