#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Everything related to lfdrs, lfdr estimation and lfdr interpolation.

@author: Martin Goelz
"""
import os
import sys
import time
import warnings

from functools import partial  # To give a second argument to mp.map

import pandas as pd
import pathos.multiprocessing as mp

import _pickle as pickle

import numpy as np
from scipy import stats

# from rbf.interpolate import RBFInterpolant

from detectors import bh

import statsmodels.api as sm
import statsmodels.tools as smtools

from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from field_handling import RadioSpatialField, RadioSpatialFieldEstimated
import smom_functions

from fdrs import smooth_fdr
from rbf.interpolate import RBFInterpolant

def apply_pounds_estimator(par_list, typ='emp'):
    """Estimate the null fraction and the alternative p-value PDF with the
    method proposed by Pounds.

    Parameters
    ----------
    typ : str
        Tells if we use a given PDF or the parameters of an MBM to compute the
        minimal value of the mixture PDF.
    par_list : list
        The list with additional parameters, depends on the typ.

    Returns
    -------
    list
        The estimated null proportion and the estimated PDF under H1.
    """
    pi0 = est_pi0_pounds(typ, par_list)

    # Underestimate the alternative component on the left hand side
    if pi0 < 1:
        f1_p = (par_list[0] - pi0)/(1-pi0)
    else:
        f1_p = np.zeros(par_list[0].size)
    if np.any(f1_p < 0):
        f1_p[np.where(f1_p < 0)] = 0

    return pi0, f1_p

def clip_lfdrs(x, x_min=0, x_max=1):
    """
    Clip given lfdr values to given range.

    Parameters
    ----------
    x : float array
        Values to be clamped onto [x_min, x_max]. lfdrs in our case.
    x_min : float, optional
        Left edge of the clipped domain. The default is 0.
    x_max : TYPE, optional
        Right edge of the clipped domain. The default is 1.

    Returns
    -------
    float array
        The clipped values

    """
    return np.clip(x, x_min, x_max)

def est_clfdrs(fd, res_path, sav_res, est_met, par_lst):
    if est_met == 'fdrs':
        max_wrk = par_lst
        num_wrk = np.min((max_wrk, os.cpu_count() - 1))  # First number for
        # server
        try:
            loaded_fdrs = pd.read_pickle(os.path.join(
                res_path, 'fdrs.pkl'))
            pi1_spa_var = loaded_fdrs['pi1_spa_var'][0][0]
            clfdrs = loaded_fdrs['clfdrs'][0]
        except FileNotFoundError:
            def single_run_fdrs(z_val, sen_idx, dim, alp_fdr=.1):
                z_vec = np.zeros(np.prod(dim))
                z_vec[sen_idx] = z_val
                z_mat = z_vec.reshape(dim)
                res_fdrs = smooth_fdr(z_mat, alp_fdr, verbose=0, missing_val=0)
                return res_fdrs 
            par_pl = mp.Pool(num_wrk)
            rtns = par_pl.starmap(partial(single_run_fdrs, dim=fd.dim),
                                  zip(fd.z[:, :], fd.sen_idx[:, :]))
            par_pl.close()
            
            pi1_spa_var = np.zeros((fd.n_MC, fd.n))
            clfdrs = np.zeros((fd.n_MC, fd.n))
            for mc in np.arange(fd.n_MC):
                pi1_spa_var[mc, :] = rtns[mc]['priors'].flatten()[
                    fd.sen_idx[mc, :]]
                clfdrs[mc, :] = 1 - rtns[mc]['posteriors'].flatten()[
                    fd.sen_idx[mc, :]]
            del rtns
            res = pd.DataFrame(
                {"pi1_spa_var": [pi1_spa_var], 
                "clfdrs": [clfdrs]
                })
            res.to_pickle(os.path.join(res_path, 'fdrs.pkl'))
        return clfdrs, 1 - pi1_spa_var
    if est_met == 'smom-sls':
        pi1_spa_var  = est_pi1_spa_var_sls(fd, res_path, sav_res, par_lst)
    elif est_met == 'smom-sns':
        pi1_spa_var  = est_pi1_spa_var_sns(fd, res_path, sav_res, par_lst)        
    else:
        print("This estimation method has not yet been implemented!")
        return None
    try:
        lfdr_res = pd.read_pickle(os.path.join(res_path, par_lst[-1] + '.pkl'))
    except FileNotFoundError:
        print("LFDRs haven't been stored for this method yet!")
        sys.exit()
    f1_p = lfdr_res['f1_p_hat'][0]
    clfdrs = np.zeros((fd.n_MC, fd.n))
    for mc in np.arange(fd.n_MC):
        clfdrs[mc, :] = (
        (1 - pi1_spa_var[mc, :]) / (
                (1 - pi1_spa_var[mc, :]) + pi1_spa_var[mc, :] * f1_p[mc, :]))
    return clfdrs, 1 - pi1_spa_var

def est_lfdrs(fd, res_path, sav_res, est_met, par_lst):
    """Estimate the lfdrs with the given estimation method.

    Parameters
    ----------
    fd : RadioSpatialFieldEstimated
        The spatial field the lfdrs shall be estimated for
    res_path : str
        The path to where the data is stored
    sav_res : boolean
        If results shall be saved
    est_met : str
        The estimation method to be used to find the lfdrs.
    par_lst : float
        Additional parameters.

    Returns
    -------
    list
        The list of quantities returned by this estimation method.
    """
    if est_met == 'lm':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_lm(
            fd, res_path, sav_res, par_lst)
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'gmm':
        lfdrs, f_p, f1_p, pi0, ex_time, num_sel_cmp = est_lfdrs_gmm(
            fd, res_path, sav_res, par_lst)
        returns = [lfdrs, f_p, f1_p, pi0, ex_time, num_sel_cmp]
    elif est_met == 'pr':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_pr(
            fd, res_path, sav_res, par_lst)
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'bum':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_bum(
            fd, res_path, sav_res, par_lst)
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'smom_s':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_smom(fd, res_path, sav_res,
                                                        par_lst, "spatial")
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'smom':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_smom(fd, res_path, sav_res,
                                                        par_lst, "random")
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'mbm-em':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_mbm_em(
            fd, res_path, sav_res, par_lst)
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'smom_s-em':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_smom_em(fd, res_path,
        sav_res, par_lst, 'spatial')
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]
    elif est_met == 'smom-em':
        lfdrs, f_p, f1_p, pi0, ex_time = est_lfdrs_smom_em(fd, res_path,
        sav_res, par_lst, 'random')
        returns = [lfdrs, f_p, f1_p, pi0, ex_time]  
    else:
        print("This estimation method has not yet been implemented!")
        returns = None
    return returns


def est_lfdrs_bum(fd, res_path, sav_res, par_list):
    """Estimate the lfdrs using the BUM.

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    lam : float
        The additional parameters, the range for both parameters.

    Returns
    -------
    numpy array
        The estimated lfdrs.
    """
    try:
        res = pd.read_pickle(os.path.join(res_path, 'bum-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        print("LM loaded!")
    except FileNotFoundError:
        print("No results found for BUM, computing ...", end="")
        [a_rge, lam_rge] = par_list  # The ranges for the grid search.

        f1_p, pi0, ex_time = est_alt_p_pdf_bum(fd, a_rge, lam_rge)
        f_p = (np.transpose(np.tile(pi0, (fd.n, 1)))
                * stats.uniform.pdf(fd.p) + (1-np.transpose(
                    np.tile(pi0, (fd.n, 1)))) * f1_p)
        lfdr = (np.transpose(np.tile(pi0, (fd.n, 1)))
        * stats.uniform.pdf(fd.p))/f_p
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "ex_time": [ex_time]
                     })
            res.to_pickle(os.path.join(res_path, 'bum-sen.pkl'))
            del res
        print("BUM completed!")
    return lfdr, f_p, f1_p, pi0, ex_time

def est_lfdrs_gmm(fd, res_path, sav_res, lam):
    """Estimate the lfdrs using a Gaussian mixture model (GMM).

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    lam : float
        The width of the central proportion used for pi0 estimation.

    Returns
    -------
    numpy array
        The estimated lfdrs.
    numpy array
        The model order of the GMMs that provided the best fit over MC runs.
    """
    gmm_com = 10  # Number of components for the GMM

    try:
        res = pd.read_pickle(os.path.join(res_path, 'gmm-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        num_sel_cmp = res['num_sel_cmp'][0]
        print("GMM loaded!")
    except FileNotFoundError:
        print("No results found for Gaussian mixture model, computing ...")
        num_sel_cmp = np.zeros(fd.n_MC)  # number of Gaussians selected in best
        # iteration. If this is often equal to gmm_com, increase gmm_com.
        ex_time = np.zeros(fd.n_MC)
        pi0 = est_pi0_efron(fd.z, mid_prp=lam)  # Estimate nul frac.

        f_z = np.zeros(fd.p.shape)
        f_p = np.zeros(fd.p.shape)
        lfdr = np.zeros(fd.p.shape)
        for mc in np.arange(0, fd.n_MC, 1):
            start_time = time.time()
            try:
                f_z[mc, :], num_sel_cmp[mc] = est_pdf(
                    fd.z[mc, :], fd.z[mc, :], fit_met="gmm", par_lst=[gmm_com])
            except ValueError:
                f_z[mc, :] = np.nan
            end_time = time.time()
            ex_time[mc] = end_time - start_time
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in divide")
                warnings.filterwarnings(
                    "ignore", message="overflow encountered in divide")
                f_p[mc, :] = (f_z[mc, :]/stats.norm.pdf(fd.z[mc, :]))
            lfdr[mc, :] = (pi0[mc] / f_p[mc, :])
            print(f'\rGMM: {mc+1}/{fd.n_MC}', end="")
            del start_time, end_time
        print('')
        lfdr[np.where(lfdr > 1)] = 1
        f1_p = np.nan + np.zeros(fd.p.shape),  # Not estimated for LM
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "ex_time": [ex_time],
                     "num_sel_cmp": [num_sel_cmp]
                     })
            res.to_pickle(os.path.join(res_path, 'gmm-sen.pkl'))
            del res
        print("Gaussian mixture model completed!")
    return lfdr, f_p, f1_p, pi0, ex_time, num_sel_cmp

def est_lfdrs_lm(fd, res_path, sav_res, lam):
    """Estimate the lfdrs using Lindsey's method.

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    lam : float
        The width of the central proportion used for pi0 estimation.

    Returns
    -------
    numpy array
        The estimated lfdrs.
    """
    try:
        res = pd.read_pickle(os.path.join(res_path, 'lm-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        print("LM loaded!")
    except FileNotFoundError:
        print("No results found for Lindseys Method, computing ...")
        ex_time = np.zeros(fd.n_MC)
        pi0 = est_pi0_efron(fd.z, mid_prp=lam)  # Estimate nul frac.

        f_z = np.zeros(fd.p.shape)
        f_p = np.zeros(fd.p.shape)
        lfdr = np.zeros(fd.p.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for mc in np.arange(0, fd.n_MC, 1):
                start_time = time.time()
                try:
                    f_z[mc, :] = est_pdf(fd.z[mc, :], fd.z[mc, :],
                                         fit_met="lm")
                except ValueError:
                    f_z[mc, :] = np.nan
                end_time = time.time()
                ex_time[mc] = end_time - start_time
                f_p[mc, :] = (f_z[mc, :]/stats.norm.pdf(fd.z[mc, :]))
                lfdr[mc, :] = (pi0[mc] / f_p[mc, :])
                print(f'\rLindseys Method: {mc+1}/{fd.n_MC}', end="")
                del start_time, end_time
        print('')
        lfdr[np.where(lfdr > 1)] = 1
        f1_p = np.nan + np.zeros(fd.p.shape),  # Not estimated for LM
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "ex_time": [ex_time]
                     })
            res.to_pickle(os.path.join(res_path, 'lm-sen.pkl'))
            del res
        print("Lindseys  Method completed!")
    return lfdr, f_p, f1_p, pi0, ex_time

def est_lfdrs_mbm_em(fd, res_path, sav_res, par_lst):
    """Estimate the lfdrs using the MBM with EM.

    Details: See [Goelz2022CISS].

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par_lst : list
        The additional parameters.

    Returns
    -------
    list
        A list with the estimated lfdrs, f_p, f1_p, pi0 and the execution time.
    """
    try:
        res = pd.read_pickle(
            os.path.join(res_path, par_lst[2] + '_mbm-em-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        print("lfdr-MBM-EM loaded!")
    except FileNotFoundError:
        print("No results found for lfdr-MBM-EM, computing ...")
        par_lst.extend([res_path, sav_res])

        f_p, a, pi, mod_ord_sel, ex_time = est_pdf(
            fd.p, fd.p, 'mbm-em', par_lst=par_lst)

        f1_p = np.zeros(f_p.shape)
        pi0 = np.zeros(fd.n_MC)

        for mc in np.arange(fd.n_MC):
            (pi0[mc], f1_p[mc, :]) = apply_pounds_estimator(
                [f_p[mc, :], a[mc, 0:mod_ord_sel[mc]],
                np.ones(mod_ord_sel[mc]),
                pi[mc, 0:mod_ord_sel[mc]], 1, mod_ord_sel[mc]], typ='mbm')

        f1_p[np.where(np.isnan(f1_p))] = np.inf

        f_p = (np.transpose(np.tile(pi0, (fd.n, 1))) * stats.uniform.pdf(fd.p)
            + (1-np.transpose(np.tile(pi0, (fd.n, 1))))
            * f1_p)
        lfdr = (
            (np.transpose(np.tile(pi0, (fd.n, 1))) * stats.uniform.pdf(fd.p))
            / f_p)
        lfdr[
            np.where(np.isnan(lfdr))] = 0
        lfdr[np.where(lfdr > 1)] = 1

        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "ex_time": [ex_time]
                     })
            res.to_pickle(
                os.path.join(res_path, par_lst[2] + '_mbm-em-sen.pkl'))
            del res
        print("lfdr-MBM-EM completed!")
    return lfdr, f_p, f1_p, pi0, ex_time

def est_lfdrs_pr(fd, res_path, sav_res, par):
    """Estimate the lfdrs using predictive recursion

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par : float
        Additional parameters: The number of permutation used for PR and the
        small non-zero value to deal with very small z-scores.

    Returns
    -------
    numpy array
        The estimated lfdrs.
    """
    try:
        res = pd.read_pickle(os.path.join(res_path, 'pr-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        print("PR loaded!")
    except FileNotFoundError:
        print("No results found for predictive recursion, computing ...")
        pr_nper, z_lim = par  # Number of passes through the data for PR + zlim
        c_fd, res_fd, ex_time = est_alt_z_pdf_pr(fd, fd.z, pr_nper, z_lim)
        f1_z = np.mean(res_fd, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in divide")
            warnings.filterwarnings(
                "ignore", message="overflow encountered in divide")
            f1_p = f1_z / stats.norm.pdf(fd.z)
        f1_p[np.where(np.isnan(f1_p))] = np.inf
        pi0 = np.mean(1 - c_fd, axis=0)
        f_p = (np.transpose(np.tile(pi0, (fd.n, 1)))
               + (1-np.transpose(np.tile(pi0, (fd.n, 1))))
               * f1_p)
        lfdr = (
            np.transpose(np.tile(pi0, (fd.n, 1))) / f_p)

        if sav_res:
            res = pd.DataFrame(
                {"f_p_hat": [f_p],
                "f1_p_hat": [f1_p],
                "lfdr_hat": [lfdr],
                "pi0_hat": [pi0],
                "ex_time": [ex_time]
                })
            res.to_pickle(os.path.join(res_path, 'pr-sen.pkl'))
            del res
        print("Predictive recursion completed!")
    return lfdr, f_p, f1_p, pi0, ex_time

def est_lfdrs_smom(fd, res_path, sav_res, par_lst, partition):
    """Estimate the lfdrs using lfdr-sMoM.

    Details: See [Goelz2022TSIPN].

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par_lst : list
        The additional parameters.
    partition : str
        The partition to be used for finding the p-value vectors.

    Returns
    -------
    list
        A list with the estimated lfdrs, f_p, f1_p, pi0 and the execution time.
    """
    try:
        if partition == 'spatial':
            res = pd.read_pickle(os.path.join(res_path, 'smom_s-sen.pkl'))
        elif partition == 'random':
            res = pd.read_pickle(os.path.join(res_path, 'smom-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        if partition == 'spatial':
            print("sMoM_s loaded!")
        elif partition == 'random':
            print("sMoM loaded!")
    except FileNotFoundError:
        [dat_path, max_wrk, par_type, quant_bits, sensoring_thr] = par_lst
        if partition == 'spatial':
            print("No results found for lfdr-sMoM_s, computing ...")
        elif partition == 'random':
            print("No results found for lfdr-sMoM, computing ...")
        # ------ Applying smom -------
        # Uncomment for for loop, useful for debugging
        # (f1_p, pi0, smom_lam, smom_a, f_p, F_p, diff_edf_est_cdf, sel_k,
        #   sel_d, ex_time) = for_loop_smom(fd, par_type, dat_path, partition,
        #                           quant_bits, sensoring_thr)
        # Uncomment for parallelization
        (f1_p, pi0, smom_lam, smom_a, f_p, F_p, diff_edf_est_cdf, sel_k, sel_d,
          ex_time) = parallel_smom(fd, par_type, dat_path, max_wrk, partition,
                                  quant_bits, sensoring_thr)

        # ------ Applying smom -------
        f1_p[np.where(np.isnan(f1_p))] = np.inf

        f_p = (np.transpose(np.tile(pi0, (fd.n, 1)))
               + (1-np.transpose(np.tile(pi0, (fd.n, 1)))) * f1_p)

        lfdr = (np.transpose(np.tile(pi0, (fd.n, 1))))/f_p
        lfdr[np.where(np.isnan(lfdr))] = 0
        lfdr[np.where(lfdr > 1)] = 1
        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "a_k_hat": [smom_a],
                     "sel_k": [sel_k],
                     "sel_d": [sel_d],
                     "pi_k_hat": [smom_lam],
                     "diff_edf_cdf": [diff_edf_est_cdf],
                     "F_p_hat": [F_p],
                     "ex_time": [ex_time]
                     })
            if partition == 'spatial':
                res.to_pickle(os.path.join(res_path, 'smom_s-sen.pkl'))
            elif partition == 'random':
                res.to_pickle(os.path.join(res_path, 'smom-sen.pkl'))
            del res
    return lfdr, f_p, f1_p, pi0, ex_time


def est_lfdrs_smom_em(fd, res_path, sav_res, par_lst, partition):
    """Estimate the lfdrs using the EM with initialization by lfdr-sMoM

    Details: See [Goelz2022CISS].

    Parameters
    ----------
    fd : RadioSpatialField or RadioSpatialFieldEstimated
        The field
    res_path : str
        The path to where the results are to be stored
    sav_res : boolean
        If the results are to be saved
    par_lst : list
        The additional parameters.

    Returns
    -------
    list
        A list with the estimated lfdrs, f_p, f1_p, pi0 and the execution time.
    """
    try:
        if partition == 'spatial':
            res = pd.read_pickle(
                os.path.join(res_path, 'smom_s-em-sen.pkl'))
        elif partition == 'random':
            res = pd.read_pickle(
                os.path.join(res_path, 'smom-em-sen.pkl'))
        lfdr = res['lfdr_hat'][0]
        f_p = res['f_p_hat'][0]
        f1_p = res['f1_p_hat'][0]
        pi0 = res['pi0_hat'][0]
        ex_time = res['ex_time'][0]
        if partition == 'spatial':
            print("lfdr-sMoM_s-EM loaded!")
        elif partition == 'random':
            print("lfdr-sMoM-EM loaded!")
    except FileNotFoundError:
        [max_wrk, cvg_thr] = par_lst
        if partition == 'spatial':
            print("No results found for lfdr-sMoM_s-EM, computing ...")
            smom_res = pd.read_pickle(os.path.join(res_path, 'smom_s-sen.pkl'))
        elif partition == 'random':
            print("No results found for lfdr-sMoM-EM, computing...")
            smom_res = pd.read_pickle(os.path.join(res_path, 'smom-sen.pkl'))

        par_lst = [smom_res["a_k_hat"][0], smom_res["pi_k_hat"][0],
                   smom_res["ex_time"][0], max_wrk, cvg_thr]

        f_p, a, pi, mod_ord_sel, ex_time, num_it = est_pdf(
            fd.p, fd.p, 'smom-em', par_lst=par_lst)

        f1_p = np.zeros(f_p.shape)
        pi0 = np.zeros(fd.n_MC)

        for mc in np.arange(fd.n_MC):
            (pi0[mc], f1_p[mc, :]) = apply_pounds_estimator(
                [f_p[mc, :], a[mc], np.ones(a[mc].shape),
                pi[mc], 1, mod_ord_sel[mc]], typ='mbm')

        f1_p[np.where(np.isnan(f1_p))] = np.inf

        f_p = (np.transpose(np.tile(pi0, (fd.n, 1))) * stats.uniform.pdf(fd.p)
            + (1-np.transpose(np.tile(pi0, (fd.n, 1))))
            * f1_p)
        lfdr = (
            (np.transpose(np.tile(pi0, (fd.n, 1))) * stats.uniform.pdf(fd.p))
            / f_p)
        lfdr[
            np.where(np.isnan(lfdr))] = 0
        lfdr[np.where(lfdr > 1)] = 1

        if sav_res:
            res = pd.DataFrame(
                    {"f_p_hat": [f_p],
                     "f1_p_hat": [f1_p],
                     "lfdr_hat": [lfdr],
                     "pi0_hat": [pi0],
                     "ex_time": [ex_time],
                     "a_k_hat": [a],
                     "pi_k_hat": [pi],
                     "num_it": [num_it]
                     })
            if partition == 'spatial':
                res.to_pickle(os.path.join(res_path, 'smom_s-em-sen.pkl'))
                print("lfdr-sMoM_s-EM completed!")
            elif partition == 'random':
                res.to_pickle(os.path.join(res_path, 'smom-em-sen.pkl'))
                print("lfdr-sMoM-EM completed!")
            del res
    return lfdr, f_p, f1_p, pi0, ex_time

def est_pi0_efron(dat, mid_prp=0.5):
    """
    Estimate the null fraction by Efron's central proportion estimator.

    Computes the fraction of z-scores by counting the number of z-scores
    within a certain central interval of width mid_prp, where mid_prp is the
    percentage of z-scores that would fall on average into this intervall,
    if the global null was true.

    Parameters
    ----------
    dat : array
        The given data array of dimension MC runs x data length.
    mid_prp : float
        Between 0 and 1, indicating the fraction of the null density to be
        considered. Larger mid_prp is more robust, but also more conservative.

    Returns
    -------
    array
        A vector iwth num MC runs entries with the estimated null proportions
        per MC run.

    @author: Martin Goelz
    """
    nul_prp = (stats.norm.cdf(stats.norm.ppf(.5 + mid_prp/2))
               - stats.norm.cdf(stats.norm.ppf(.5 - mid_prp/2)))
    jnt_num = np.sum(np.all((dat <= stats.norm.ppf(.5 + mid_prp/2),
                             dat >= stats.norm.ppf(.5 - mid_prp/2)), axis=0),
                     axis=1)
    pi0 = jnt_num/(dat.shape[1] * nul_prp)
    return np.min((pi0, np.ones(pi0.shape[0])), axis=0)

def est_pi0_pounds(typ, par_list):
    """Estimate the null fraction with the method proposed by Pounds.

    Parameters
    ----------
    typ : str
        Tells if we use a given PDF or the parameters of an MBM to compute the
        minimal value of the mixture PDF.
    par_list : list
        The list with additional parameters, depends on the typ.

    Returns
    -------
    float
        The estimated null proportion.
    """
    if typ == 'mbm':
        [_, a, b, w, d, k] = par_list
        if a.ndim == 1:
            a = a[:, np.newaxis]
            b = b[:, np.newaxis]
        d = int(d)
        k = int(k)
        # Overestimate the alternative proportion
        p_grid = np.arange(1/1000, 1, 1/1000)
        # Check if this is an averaged parameter model or not
        pdf = get_pdf_multivariate_mbm(
                p_grid, a[0:k, 0:d], b[0:k, 0:d], w[0:k], d, k)
        pi0 = np.nanmin(pdf)
    elif typ == 'emp':
        pdf = par_list
        pi0 = np.max((0, np.nanmin(pdf)))
    return pi0

def est_pi1_spa_var_sls(fd, res_path, sav_res, par_lst):
    dat_path, max_wrk, par, base_str = par_lst
    try:
        lfdr_res = pd.read_pickle(os.path.join(res_path, base_str + '.pkl'))
    except FileNotFoundError:
        print("LFDRs haven't been stored for this method yet!")
        sys.exit()
    try:
        res = pd.read_pickle(
            os.path.join(res_path, base_str + '-sls.pkl'))
        pi1_spa_var_opt = res['pi1_spa_var'][0]
        print("clfdr-" + base_str + "-SLS loaded!")
    except FileNotFoundError:
        with (open(
            os.path.join(dat_path, "..", "spa_var_par_") + par + '.pkl', 'rb')
                as input):
            loaded = pickle.load(input)
        bw_grid = loaded[0]
        ker_vec = loaded[1]
        pi0_max = loaded[3]
        lfdrs = lfdr_res['lfdr_hat'][0]
        f1_p = lfdr_res['f1_p_hat'][0]
        f1_z = f1_p * stats.norm.pdf(fd.z)

        num_wrk = np.min((max_wrk, os.cpu_count() - 1))  # First number for
        pi1_spa_var = np.zeros((len(ker_vec), bw_grid.size, fd.n_MC, fd.n))

        for (ker_idx, ker) in enumerate(ker_vec):
            for (kbw_idx, kbw_val) in enumerate(bw_grid):
                print(
                    f"\rclfdr-{base_str}-SLS for Kernel {ker} bandwidth grid"
                    f"search running: {kbw_idx + 1:2} / "
                    f"{bw_grid.size:2}", end="")
                par_pl = mp.Pool(num_wrk)
                rtns_lfdr_knor = par_pl.starmap(
                    partial(
                        single_run_sls, kernel=ker, h=kbw_val),
                    zip(lfdrs, fd.sen_cds))
                par_pl.close()
                for mc in np.arange(fd.n_MC):
                    pi1_spa_var[ker_idx, kbw_idx, mc, :] = rtns_lfdr_knor[mc]
        print("")
        print(f"\rclfdr-{base_str}-SLS completed!")
        obj = np.zeros((fd.n_MC, len(ker_vec), bw_grid.size))
        for (ker_idx, ker) in enumerate(ker_vec):
            for (kbw_idx, kbw_val) in enumerate(bw_grid):
                for mc in np.arange(fd.n_MC):
                    obj[mc, ker_idx, kbw_idx] = (np.nansum(
                        (pi1_spa_var[ker_idx, kbw_idx, mc, :] * f1_z[mc, :] +
                         (1 - pi1_spa_var[ker_idx, kbw_idx, mc, :])
                         * stats.norm.pdf(fd.z[mc, :])))/fd.n)
        pi1_spa_var = np.nan_to_num(pi1_spa_var, nan=1 - pi0_max)
        opt_idx = np.zeros((fd.n_MC, 2), dtype=int)
        pi1_spa_var_opt = np.zeros((fd.n_MC, fd.n))
        for mc in np.arange(fd.n_MC):
            opt_idx[mc, :] = np.unravel_index(
                np.argmax(
                    obj[mc, :, :]), (len(ker_vec), bw_grid.size))
            opt_idx1, opt_idx2 = opt_idx[mc, :]
            pi1_spa_var_opt[mc, :] = np.squeeze(
                pi1_spa_var[opt_idx1, opt_idx2, mc, :])

        if sav_res:
            res = pd.DataFrame(
                {"pi1_spa_var": [pi1_spa_var_opt]})
            res.to_pickle(os.path.join(res_path, base_str + '-sls.pkl'))
    return pi1_spa_var_opt

def est_pi1_spa_var_sns(fd, res_path, sav_res, par_lst):
    dat_path, max_wrk, par, base_str = par_lst
    try:
        lfdr_res = pd.read_pickle(os.path.join(res_path, base_str + '.pkl'))
    except FileNotFoundError:
        print("LFDRs haven't been stored for this method yet!")
        sys.exit()
    try:
        res = pd.read_pickle(
            os.path.join(res_path, base_str + '-sns.pkl'))
        pi1_spa_var_opt = res['pi1_spa_var'][0]
        print("clfdr-" + base_str + "-SNS loaded!")
    except FileNotFoundError:
        with (open(
            os.path.join(dat_path, "..", "spa_var_par_") + par + '.pkl', 'rb')
            as input):
            loaded = pickle.load(input)
        bw_grid = loaded[0]
        ker_vec = loaded[1]
        laws_sthr = loaded[2]
        pi0_max = loaded[3]
        f1_p = lfdr_res['f1_p_hat'][0]
        f1_z = f1_p * stats.norm.pdf(fd.z)

        num_wrk = np.min((max_wrk, os.cpu_count() - 1))  # First number for
        pi1_spa_var = np.zeros((len(ker_vec), bw_grid.size, laws_sthr.size,
                                fd.n_MC, fd.n))

        for (ker_idx, ker) in enumerate(ker_vec):
            for (kbw_idx, kbw_val) in enumerate(bw_grid):
                for (sthr_idx, sthr_val) in enumerate(laws_sthr):
                    print(
                        f"\rclfdr-{base_str}-SNS for Kernel {ker} bandwidth "
                        f"gridsearch running: {kbw_idx + 1:2} "
                        f"/ {bw_grid.size:2} and "
                        f"{sthr_idx + 1:2} / {laws_sthr.size:2}",
                        end="")
                    par_pl = mp.Pool(num_wrk)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        rtns = par_pl.starmap(
                            partial(
                                single_run_sns, siz=fd.dim, kernel=ker,
                                laws_sthr=sthr_val, h=kbw_val,
                                exclude_sen=True),
                            zip(fd.p, fd.sen_idx))
                    par_pl.close()
                    for mc in np.arange(fd.n_MC):
                        aux = rtns[mc]
                        pi1_spa_var[ker_idx, kbw_idx, sthr_idx, mc, :] = aux[
                            fd.sen_idx[mc, :]]
                    del aux
        print("")
        print(f"\rclfdr-{base_str}-SNS completed!")
        obj = np.zeros((fd.n_MC, len(ker_vec), bw_grid.size, laws_sthr.size))
        for (ker_idx, ker) in enumerate(ker_vec):
            for (kbw_idx, kbw_val) in enumerate(bw_grid):
                for (sthr_idx, sthr_val) in enumerate(laws_sthr):
                    for mc in np.arange(fd.n_MC):
                        obj[mc, ker_idx, kbw_idx, sthr_idx] = (np.nansum(
                                (pi1_spa_var[ker_idx,
                                    kbw_idx, sthr_idx, mc, :] * f1_z[mc, :] +
                                    (1 - pi1_spa_var[ker_idx,
                                        kbw_idx, sthr_idx, mc, :])
                                    * stats.norm.pdf(fd.z[mc, :])))/fd.n)

        pi1_spa_var = np.nan_to_num(pi1_spa_var, nan=1-pi0_max)

        opt_idx = np.zeros((fd.n_MC, 3), dtype=int)
        pi1_spa_var_opt = np.zeros((fd.n_MC, fd.n))
        for mc in np.arange(fd.n_MC):
            opt_idx[mc, :] = np.unravel_index(
                np.argmax(obj[mc, :, :, :]),
                (len(ker_vec), bw_grid.size, laws_sthr.size))
            opt_idx1, opt_idx2, opt_idx3 = opt_idx[mc, :]
            pi1_spa_var_opt[mc, :] = np.squeeze(
                pi1_spa_var[opt_idx1, opt_idx2, opt_idx3, mc, :])

        if sav_res:
            res = pd.DataFrame(
                {"pi1_spa_var": [pi1_spa_var_opt]
                })
            res.to_pickle(os.path.join(res_path, base_str + '-sns.pkl'))
    return pi1_spa_var_opt

def est_pdf(dat, evl_dat, fit_met, par_lst=[7, .05, "irls"]):
    """
    Compute the pdf of data using a given fitting method.

    Possible methods:
        - "lm": Lindsey's method [Efron and Tibshirani1996].
            The standard according to Efron's book, because it creates quite
            smooth estimates with hence low variance. However, can be heavily
            biased if the estimated pdf is not smooth/difficult to approximate
            with.
            Use Poisson regression to estimate the parameters of an exponential
            family distribution. Returns the estimated pdf at the given
            evaluation data points and the estimated distribution parameters
            (if desired). The resulting estimated PDF is the MLE for the true
            pdf at the bin centers.
        - "gmm": Gaussian mixture model.
            Classic parametric approach for PDF estimation. The only tuning
            parameter is the number of components. So far, we consider the
            number of components as fixed and hand it over in the parameter
            list.
        - "mbm-em": MBM-EM
            A multi-single beta parameter mixture model (MBM) with its params
            estimated by expectation maximiziation (EM). See [Goelz2022CISS]
            for details.

    Parameters
    ----------
    dat : array
        The input data.
    evl_dat: array
        The data at which the pdf is to be evaluated
    fit_met : str, optional
        The method to estimate the pdf.

    Returns
    -------
    numpy array
        The estimated pdf values at the given evaluation data.

    @author: Martin Goelz
    """
    if fit_met == "lm":
        J, bin_width, method = par_lst
        bin_lim = np.max(np.abs([np.min(dat), np.max(dat)])) + 1
        bin_edg = np.arange(-bin_lim, bin_lim+bin_width, bin_width)

        # Binning the data
        y, _ = np.histogram(dat, bins=bin_edg)

        # The bin centers
        bin_cen = bin_edg[0:-1] + bin_width/2

        # Auxiliary matrix to create the polynomial expansion matrix
        pow_mat = np.transpose(
            np.tile(np.arange(0, J, 1) + 1, (bin_cen.size, 1)))

        # The polynomial expansion matrix
        M = np.transpose(np.power(np.tile(bin_cen, (J, 1)), pow_mat))

        # GLM fitting by Poisson regression
        glm_mod = sm.GLM(
            y, smtools.add_constant(M), family=sm.families.Poisson())
        try:
            glm_res = glm_mod.fit(method=method)  # Default fitting: IRLS
            # (iteratively reweighted
        except ValueError:
            glm_res = glm_mod.fit(method='lbfgs')  # If IRLS fails, do direct
            # optimization, which is a bit more stable
        # least squares)
        beta_hat = glm_res.params  # The MLEs for the exponential family dist.

        # Estimation of PDF values at bin centers
        x_vec = evl_dat  # Data points at which estimated density is
        # computed

        # Auxiliary matrix with polynomials in the argument of Eq. (5.10)
        # [Efron2011]
        pow_mat = np.transpose(np.tile(np.arange(0, J + 1, 1),
                                       (x_vec.size, 1)))

        # The data point matrix containing the points at which the pdf is
        # to be evaluated at
        x_mat = np.transpose(np.power(np.tile(x_vec, (J + 1, 1)), pow_mat))

        # estimated pdf
        arg = np.matmul(x_mat, beta_hat)
        pdf_hat = np.exp(arg)/dat.size/bin_width  # Normalization for a
        # valid PDF

        return pdf_hat

    elif fit_met == "gmm":
        [ncom_max] = par_lst
        candidate_gmms = {}
        bic = np.zeros(ncom_max)
        bic = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for (k_idx, k) in enumerate(np.arange(1, ncom_max + 1, 1)):
                candidate_gmms[k] = (GaussianMixture(n_components=k))
                resi = candidate_gmms[k].fit(dat.reshape((len(dat), 1)))
                bic[k] = candidate_gmms[k].bic(dat.reshape(-1, 1))
        k_opt = min(bic, key=bic.get)
        return [
            np.exp(candidate_gmms[k_opt].score_samples(evl_dat.reshape(
                (len(evl_dat), 1)))), k_opt]

    elif fit_met == 'mbm-em':
        dat_path, max_wrk, par_type, res_path, sav_res = par_lst
        try:
            loaded = pd.read_pickle(
                os.path.join(res_path, par_type + '_mbm-em_est_par.pkl'))
            mod_ord_sel = loaded['mod_ord_sel'][0]
            a_k = loaded['a_k'][0]
            pi_k = loaded['pi_k'][0]
            num_iter = loaded['num_iter'][0]
            bic = loaded['bic'][0]
            ex_time = loaded['ex_time'][0]
            n_MC = mod_ord_sel.size
        except FileNotFoundError:
            with open(os.path.join(dat_path, "..", "mbm-em_par_")
                + par_type + '.pkl', 'rb') as input:
                ld_par = pickle.load(input)
                [mod_ords, n_reps, cvg_thr] = [x for x in ld_par]
            del ld_par

            n_MC = dat.shape[0]
            mod_ord_sel = np.zeros(n_MC, dtype=int)
            a_k = np.zeros((n_MC, mod_ords.size))
            pi_k = np.zeros((n_MC, mod_ords.size))
            num_iter = np.zeros(n_MC)
            bic = np.zeros((n_MC, mod_ords.size))
            ex_time = np.zeros(n_MC)

            num_wrk = np.min((max_wrk, os.cpu_count() - 1))
            par_pl = mp.Pool(num_wrk)
            rtns = par_pl.map(partial(
                    single_run_mbm_em, mod_ords=mod_ords, n_reps=n_reps,
                    cvg_thr=cvg_thr), dat)

            par_pl.close()
            for mc in np.arange(n_MC):
                mod_ord_sel[mc] = rtns[mc][0]
                a_k[mc, 0:mod_ord_sel[mc]] = rtns[mc][1][:]
                pi_k[mc, 0:mod_ord_sel[mc]] = rtns[mc][2][:]
                num_iter[mc] = rtns[mc][3]
                bic[mc] = rtns[mc][4]
                ex_time[mc] = rtns[mc][5]

            # for mc in np.arange(n_MC):
            #     [mod_ord_sel[mc], a_k_tmp, pi_k_tmp, num_iter[mc], bic[mc],
            #       ex_time[mc]] = single_run_mbm_em(
            #         dat[mc, :], mod_ords, n_reps, cvg_thr)
            #     a_k[mc, 0:mod_ord_sel[mc]] = a_k_tmp
            #     pi_k[mc, 0:mod_ord_sel[mc]] = pi_k_tmp

            if sav_res:
                res = pd.DataFrame(
                    {"mod_ord_sel": [mod_ord_sel],
                     "a_k": [a_k],
                     "pi_k": [pi_k],
                     "num_iter": [num_iter],
                     "bic": [bic],
                     "ex_time": [ex_time]
                     })
                res.to_pickle(os.path.join(
                    res_path, par_type + '_mbm-em_est_par.pkl'))

        pdf_hat = np.zeros(dat.shape)
        # Segmentation fault may occur for super small p-values below ~1e-305.
        dat[np.where(dat<1e-305)] = 1e-305
        for mc in np.arange(n_MC):
            pdf_k_hat = np.zeros((mod_ord_sel[mc], dat.shape[1]))
            for k in np.arange(mod_ord_sel[mc]):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    pdf_k_hat[k, :] = stats.beta.pdf(dat[mc, :], a_k[mc, k], 1)
                pdf_hat[mc, :] = pdf_hat[mc, :] + pi_k[mc, k] * pdf_k_hat[k, :]
        return pdf_hat, a_k, pi_k, mod_ord_sel, ex_time

    elif fit_met == 'smom-em':
        a0, pi0, ex_time_init, max_wrk, cvg_thr = par_lst
        
        cvg_thr_exp = np.zeros(pi0.shape[0]) + cvg_thr
        num_wrk = np.min((max_wrk, os.cpu_count() - 1))


        pdf_hat = np.zeros(dat.shape)
        pi_k = []
        a_k = []
        mod_ord_sel = np.zeros(pi0.shape[0], dtype=int)
        ex_time = np.zeros(pi0.shape[0])
        num_it = np.zeros(pi0.shape[0], dtype=int)

        par_pl = mp.Pool(num_wrk)
        rtns = par_pl.starmap(single_run_smom_em,
                              zip(dat, a0, pi0, cvg_thr_exp))
        par_pl.close()

        for mc in np.arange(pi0.shape[0]):
            pdf_hat[mc, :] = rtns[mc][0][:]
            mod_ord_sel[mc] = rtns[mc][1]
            pi_k.append(rtns[mc][2])
            a_k.append(rtns[mc][3])
            ex_time[mc] = rtns[mc][4] + ex_time_init[mc]
            num_it[mc] = rtns[mc][5]

        # for mc in np.arange(pi0.shape[0]):
        #     [pdf_hat[mc, :], mod_ord_sel[mc], pi_k_tmp, a_k_tmp,
        #      ex_time[mc], num_it[mc]] = single_run_smom_em(
        #          dat[mc, :], a0[mc, :, :], pi0[mc, :], cvg_thr_exp[mc])
        #     a_k.append(a_k_tmp)
        #     pi_k.append(pi_k_tmp)


        return pdf_hat, a_k, pi_k, mod_ord_sel, ex_time, num_it

def est_alt_p_pdf_bum(fd, a_rge, lam_rge):
    """Estimate the alternative p-value PDF using BUM.

    Parameters
    ----------
    fd : RadioSpatialField/RadioSpatialFieldEstimated
        The field the pdf is to be estimated for.
    a_rge : numpy array
        The range of possible values of the beta shape parameter.
    lam_rge : numpy array
        The range of possible values of the uniform weight parameter.
    """
    def bum_fit(p, a_rge, lam_rge):
        llf = np.zeros(shape=(a_rge.size, lam_rge.size))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for a_idx in np.arange(0, a_rge.size, 1):
                for lam_idx in np.arange(0, lam_rge.size, 1):
                    llf[a_idx, lam_idx] = np.sum(
                        np.log(lam_rge[lam_idx]
                               + (1-lam_rge[lam_idx])*a_rge[a_idx]
                               * (p**(a_rge[a_idx]-1))))
        a_hat_idx, lam_hat_idx = np.unravel_index(np.nanargmax(llf), llf.shape)
        a_hat = a_rge[a_hat_idx]
        lam_hat = lam_rge[lam_hat_idx]
        return a_hat, lam_hat, llf

    pdf_mix = np.zeros(fd.p.shape)
    f1_p = np.zeros(fd.p.shape)
    pi_0_hat = np.zeros(fd.p.shape[0])
    ex_time = np.zeros(fd.p.shape[0])
    for mc in np.arange(fd.n_MC):
        start_time = time.time()
        idx_vec = fd.p[mc, :] > -np.inf  # 1e-305
        beta_a, beta_lam, _ = bum_fit(fd.p[mc, idx_vec], a_rge, lam_rge)
        beta_lam = np.min((beta_lam, 1))
        # Computing the mixture pdf values all z-scores
        pdf_mix[mc, idx_vec] = get_pdf_bum(fd.p[mc, idx_vec], beta_a, beta_lam)
        # Estimation as proposed by [Pounds2003]
        pi_0_hat[mc], f1_p[mc, idx_vec] = apply_pounds_estimator(
            [pdf_mix[mc, idx_vec]])
        end_time = time.time()
        ex_time[mc] = end_time - start_time
    return f1_p, pi_0_hat, ex_time

def est_alt_z_pdf_pr(fd, vls, nper, z_lim):
    """Estimate the alternative z-score density by predictive recursion.

    The principle is to estimate the prior distribution of the centrality
    parameter and place a point mass at z = 0 for the null component.

    Parameters
    ----------
    fd : RadioSpatialField
        The field the density is to be compuuted for
    vls : array
        The values pdf values are to be returned for. Either a vector, then the
        same values are taken for all MC runs, or the number of rows are equal
        to the number of MC runs in fd.
    nper : int
        The number of permutations of the data = number of times we pass
        through the data. A low number of passes in the range of 10 or 20
        is typically sufficient
    z_lim : float
        Non-zero constant to deal with really really small values.

    Returns
    -------
    c : array
        The estimated alternative proportion as an nper x MC array.
    pi : array
        The estimated prior.
    res : array
        The estimated pdf values.

        @author: Martin Goelz
    """
    def sgl_mc_run(mc):
        start_time = time.time()
        c = np.zeros(nper)
        pi = np.zeros((nper, theta.size))
        res = np.zeros((nper, vls.shape[1]))
        idx_vec = dat[mc, :] > z_lim
        for per in np.arange(0, nper, 1):
            pi_0 = 0.5
            pi_1_til = (1-pi_0)*stats.uniform.pdf(
                theta, loc=np.min(dat[mc, idx_vec]),
                scale=(np.max(dat[mc, idx_vec])-np.min(dat[mc, idx_vec])))
            per_idx = np.random.choice(dat[mc, idx_vec].size,
                                       dat[mc, idx_vec].size,
                                       replace=False)
            for i in np.arange(dat[mc, idx_vec].size):
                m_0 = pi_0*stats.norm.pdf(dat[mc, idx_vec][per_idx[i]])
                f_1 = stats.norm.pdf(
                    dat[mc, idx_vec][per_idx[i]], loc=theta)*pi_1_til
                m_1 = np.trapz(f_1, x=theta)
                pi_0 = (1-wei[i])*pi_0 + wei[i]*(m_0/(m_0+m_1))
                pi_1_til = (1-wei[i])*pi_1_til + wei[i]*(f_1/(m_0+m_1))

            c[per] = 1-pi_0
            pi[per, :] = pi_1_til/c[per]
            for i in np.arange(dat[mc, :].size):
                res[per, i] = np.trapz(
                    stats.norm.pdf(dat[mc, i], loc=theta)*pi[per, :], x=theta)
        print(f"\rFinished MC run {mc+1}/{MC}", end="")
        end_time = time.time()
        ex_time = end_time - start_time
        return c, res, ex_time

    if vls.ndim == 1:
        vls = np.tile(vls, (fd.n_MC, 1))
    dat = fd.z
    n = fd.n
    MC = fd.n_MC
    theta = np.arange(np.nanmin(dat), np.nanmax(dat), 0.01)
    a = .67
    wei = (np.arange(0, n, 1) + 2)**(-a)

    c = np.zeros((nper, MC))
    res = np.zeros((nper, MC, vls.shape[1]))
    ex_time = np.zeros((MC))
    num_wrk = np.min((50, os.cpu_count() - 1))  # First number for
    # server.
    par_pl = mp.Pool(num_wrk)
    rtns = par_pl.map(sgl_mc_run, np.arange(0, MC, 1))
    for mc in np.arange(0, MC, 1):
        c[:, mc] = rtns[mc][0][:]
        res[:, mc, :] = rtns[mc][1][:]
        ex_time[mc] = rtns[mc][2]
    par_pl.close()
    # for mc in np.arange(0, fd.n_MC, 1):
    #     (c[:, mc] ,
    #         res[:, mc, :], ex_time[mc]) = sgl_mc_run(mc)

    return c, res, ex_time

def for_loop_smom(fd, par_type, dat_path, partition, quant_bits,
                  sensoring_thr):
    """Estimate the alternative p-value PDF for lfdr-sMoM with a for loop (for
    debugging.)

    Parameters
    ----------
    fd : RadioSpatialField/RadioSpatialFieldEstimated
        The field the lfdrs are to be estimated for.
    par_type : str
        The smom parametrization type. Recommended: 'stan' for standard.
    dat_path : str
        The path to where the data is stored.
    partition : str
        The type of partition for obtaining the p-value vectors.
    quant_bits : int or None
        The number of bits used for quantization.
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring occurs.

    Returns
    -------
    list
        The list with all estimated quantities.
    """
    with open(os.path.join(dat_path, "..", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        if partition == 'spatial':
            mom_d = mom_d[
                np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
        del ld_par

    # Setting up null proportion and alternative density
    pi_0_hat = np.zeros((fd.n_MC))
    f1_hat = np.zeros((fd.n_MC, fd.n))

    # Setting up the variables for storing the results
    mom_lam_hats = np.zeros((fd.n_MC, np.max(mom_k)))
    mom_a_hats = np.zeros((fd.n_MC, np.max(mom_k), np.max(mom_d)))
    mom_p_pdf = np.zeros((fd.n_MC, fd.n))
    mom_p_cdf = np.zeros((fd.n_MC, fd.n))
    mom_sel_k = np.zeros((fd.n_MC), dtype=int)
    mom_sel_d = np.zeros((fd.n_MC), dtype=int)
    ex_time = np.zeros(fd.n_MC)
    mom_diff_best = np.zeros(fd.n_MC)
    if partition == 'spatial':
        for mc in np.arange(0, fd.n_MC, 1):
            (mom_a_hats[mc, :, :],
                mom_lam_hats[mc, :], mom_p_pdf[mc, :],
                mom_p_cdf[mc, :], mom_diff_best[mc],
                mom_sel_k[mc], mom_sel_d[mc],
                ex_time[mc])  = single_run_smom_spatial(
                    fd.p[mc, :], dat_path, par_type, quant_bits,
                    sensoring_thr)
            # Estimating the alternative density and null proportion by Pounds
            # method
            (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
                [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
                np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
                mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
            print(f"\rFinished MC run {mc+1}/{fd.n_MC}", end="")
    elif partition == 'random':
        for mc in np.arange(0, fd.n_MC, 1):
            (mom_a_hats[mc, :, :],
                mom_lam_hats[mc, :], mom_p_pdf[mc, :],
                mom_p_cdf[mc, :], mom_diff_best[mc],
                mom_sel_k[mc], mom_sel_d[mc],
                ex_time[mc]) = single_run_smom_random(fd.p[mc, :], dat_path,
                par_type, quant_bits, sensoring_thr)
            # Estimating the alternative density and null proportion by Pounds
            # method
            (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
                [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
                np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
                mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
            print(f"\rFinished MC run {mc+1}/{fd.n_MC}", end="")
    print("")
    return (f1_hat, pi_0_hat, mom_lam_hats, mom_a_hats,
            mom_p_pdf, mom_p_cdf, mom_diff_best, mom_sel_k, mom_sel_d,
            ex_time)

def get_cdf_multivariate_mbm(dat, a, b, w, d, k):
    """Returns the CDF values at given data points for a multivariate
    multi-parameter beta distribution model (MBM) with the given
    parameters.

    Parameters
    ----------
    dat : numpy array
        A one-dimensional numpy array with points where the CDF shall be
        evaluated.
    a : numpy array
        A k x d numpy array with the first beta shape parameter.
    b : numpy array
        A k x d numpy array with the second beta shape parameter. We use b = 1
        in the present works.
    w : numpy array
        A one-dimensional (length k) numpy array with the weights for each
        multivariate component
    d : int
        The multivariate dimension.
    k : int
        The number of mixture components.

    Returns
    -------
    numpy array
        The CDF values evaluated at the data in dat.
    """
    cdf = np.zeros((k, d, dat.size))
    use_k_idc = is_k_ok(w)
    use_d_idc = np.ones(a.shape)
    use_d_idc[~use_k_idc] = 0
    for k_idx in np.arange(0, k, 1):
        if use_k_idc[k_idx]:
            for d_idx in np.arange(0, d, 1):
                use_d_idc[k_idx, d_idx] = is_d_ok(
                    a[k_idx, d_idx], b[k_idx, d_idx])
                if use_d_idc[k_idx, d_idx]:
                    cdf[k_idx, d_idx, :] = (stats.beta.cdf(
                            dat, a[k_idx, d_idx], b[k_idx, d_idx]))
                else:
                    cdf[k_idx, d_idx, :] = np.nan
        else:
            cdf[k_idx, :, :] = np.nan
            use_d_idc[k_idx, :] = 0
    use_k_idc[np.where(use_k_idc)[0]] = np.sum(
        use_d_idc[np.where(use_k_idc)[0], :], 1)>0
    cdf_av = np.zeros((d, dat.size))
    w_tmp = np.copy(w)
    w_tmp[~use_k_idc] = 0
    w_tmp = w_tmp/np.sum(w_tmp)
    for k_idx in np.where(use_k_idc)[0]:
        cdf_av[np.where(
            use_d_idc[k_idx, :])[0], :] = (cdf_av[np.where(
            use_d_idc[k_idx, :])[0], :]  + w_tmp[k_idx] *
                1/np.sum(use_d_idc[k_idx, :]) * cdf[k_idx, np.where(
            use_d_idc[k_idx, :])[0], :])
    cdf_av = np.sum(cdf_av, axis=0)
    return cdf_av

def get_pdf_bum(val, a, lam):
    """Return the PDF values for the BUM model with given parameters.

    Parameters
    ----------
    val : numpy array
        The values the PDF is to be computed for.
    a : float
        The beta distribution shape parameter
    lam : float
        The weight for the uniform component (between 0 and 1)

    Returns
    -------
    numpy array
        The PDF values.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return (lam + (1-lam)*a*(val**(a-1)))

def get_pdf_multivariate_mbm(dat, a, b, w, d, k):
    """Returns the PDF values at given data points for a multivariate
    multi-parameter beta distribution model (MBM) with the given
    parameters.

    Parameters
    ----------
    dat : numpy array
        A one-dimensional numpy array with points where the PDF shall be
        evaluated.
    a : numpy array
        A k x d numpy array with the first beta shape parameter.
    b : numpy array
        A k x d numpy array with the second beta shape parameter. We use b = 1
        in the present works.
    w : numpy array
        A one-dimensional (length k) numpy array with the weights for each
        multivariate component
    d : int
        The multivariate dimension.
    k : int
        The number of mixture components.

    Returns
    -------
    numpy array
        The PDF values evaluated at the data in dat.
    """
    pdf = np.zeros((k, d, dat.size))
    use_k_idc = is_k_ok(w)
    use_d_idc = np.ones(a.shape)
    use_d_idc[~use_k_idc] = 0
    # Segmentation fault may occur for super small p-values below ~1e-305.
    dat[np.where(dat<1e-305)] = 1e-305
    for k_idx in np.arange(0, k, 1):
        if use_k_idc[k_idx]:
            for d_idx in np.arange(0, d, 1):
                use_d_idc[k_idx, d_idx] = is_d_ok(
                    a[k_idx, d_idx], b[k_idx, d_idx])
                if use_d_idc[k_idx, d_idx]:
                    pdf[k_idx, d_idx] = (stats.beta.pdf(
                        dat, a[k_idx, d_idx],
                        b[k_idx, d_idx]))
                else:
                    pdf[k_idx, d_idx, :] = np.nan
        else:
            pdf[k_idx, :, :] = np.nan
            use_d_idc[k_idx, :] = 0
    use_k_idc[np.where(use_k_idc)[0]] = np.sum(
        use_d_idc[np.where(use_k_idc)[0], :], 1) > 0
    pdf_av = np.zeros((d, dat.size))
    w_tmp = np.copy(w)
    w_tmp[~use_k_idc] = 0
    w_tmp = w_tmp/np.sum(w_tmp)
    for k_idx in np.where(use_k_idc)[0]:
        pdf_av[np.where(
            use_d_idc[k_idx, :])[0], :] = (pdf_av[np.where(
                use_d_idc[k_idx, :])[0], :] + w_tmp[k_idx] *
                1/np.sum(use_d_idc[k_idx, :]) * pdf[k_idx, np.where(
                    use_d_idc[k_idx, :])[0], :])
    # pdf_av = np.mean(pdf_av, axis=0)
    pdf_av = np.sum(pdf_av, axis=0)
    return pdf_av

def get_p_alt_pdf_jnt(fd, vls):
    """
    Compute the true alternate joint p-value pdf values for given p-values.

    Joint pdf of all alternative spatial units in the field, obtained as a
    weighted sum of the individial local pdfs. Works only for energy detectors.
    If other test statistics are used, this has to be fixed.

    Parameters
    ----------
    fd : RadioSpatialField
        The field to be analyzed
    vls : array
        The array with the values the pdf is to be computed for. Can be a
        vector or matrix, where in the latter case, the number of rows is
        equal to number of MC runs in the field.

    Returns
    -------
    alt_pdf_jnt : array
        The matrix with the alternative pdf values. One row per MC run,
        because different alternative pdf might be valid in different MC runs
        stored in the field. The columns are always the same, as they all
        correspond to the respective value in the input data vector.

    @author: Martin Goelz
    """
    if vls.ndim == 1:
        vls = np.tile(vls, (fd.n_MC, 1))

    # Defining all required quantities locally! To not kill the RAM
    n1 = fd.n_1
    try:
        n_obs = fd.n_obs_per_sen
    except:
        n_obs = fd.n_obs + np.zeros((fd.n_MC, fd.n))
    X = fd.X
    r_tru = fd.r_tru
    MC = fd.n_MC

    def do_p_alt_pdf_jnt(mc):
        """
        Compute the alternate joint p-value pdf for one mc run.

        Parameters
        ----------
        mc : int
            The MC run

        Returns
        -------
        alt_pdf_jnt_mc : array
            The alternate pdf values.

        @author: Martin Goelz
        """
        n_obs_true_alt = n_obs[mc, r_tru[mc, :]]
        nc_alt = n_obs_true_alt * X[mc, r_tru[mc, :]]**2
        alt_pdf_cmp = np.zeros((n1[mc], vls.shape[1]))
        for ind_idx in np.arange(n1[mc]):
            alt_pdf_cmp[ind_idx, :] = 1/n1[mc]/stats.chi2.pdf(
                stats.chi2.ppf(1-vls[mc, :], df=n_obs_true_alt[ind_idx]),
                df=n_obs_true_alt[ind_idx])*stats.ncx2.pdf(
                    stats.chi2.ppf(1-vls[mc, :], df=n_obs_true_alt[ind_idx]),
                    df=n_obs_true_alt[ind_idx], nc=nc_alt[ind_idx])
        # del vls_mc, nc_alt_mc
        alt_pdf_jnt_mc = np.sum(alt_pdf_cmp, axis=0)
        print(f'\rFinished MC run {mc+1}/{MC}', end="")
        return alt_pdf_jnt_mc

    num_wrk = np.min((50, os.cpu_count() - 1))  # First number for server.
    print(f"Starting a parallel pool with {num_wrk} workers.")
    par_pl = mp.Pool(num_wrk)
    rtns = par_pl.map(do_p_alt_pdf_jnt, np.arange(0, MC, 1))
    par_pl.close()
    alt_pdf_jnt = np.zeros(vls.shape)
    for mc in np.arange(0, fd.n_MC, 1):
        alt_pdf_jnt[mc, :] = rtns[mc][:]

    # alt_pdf_jnt = np.zeros(vls.shape)
    # for mc in np.arange(0, fd.n_MC, 1):
    #     try:
    #         alt_pdf_jnt[mc, :] = do_p_alt_pdf_jnt(mc)
    #     except:
    #         print('Something went wrong while calculating the alt p val'
    #                     ' pdf')
    alt_pdf_jnt[np.where(np.isnan(alt_pdf_jnt))] = 1e20  # Replacing nans by a
    # large positive constant.
    return alt_pdf_jnt

def get_true_lfdrs(fd, res_path, save_res):
    """Compute or load the ground true lfdrs.

    Parameters
    ----------
    fd : RadioSpatialField
        The field.
    res_path : str
        The path to the results
    save_res : boolean
        If results shall be saved.
    """
    if isinstance(fd, RadioSpatialField):
        try:
            print('\rRead in the ground truth lfdrs...', end="")
            loaded = pd.read_pickle(
                os.path.join(res_path, 'ground-truth.pkl'))
            lfdr = loaded['lfdr'][0]
            f1_p = loaded['f1_p'][0]
            f_p = loaded['f_p'][0]
            print('completed!')

        except FileNotFoundError:
            print('\rCompute the ground truth lfdrs...', end="")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                f1_p = get_p_alt_pdf_jnt(fd, fd.p)

            f_p = (np.transpose(np.tile(fd.pi0, (fd.n, 1)))
                   + (1-np.transpose(np.tile(fd.pi0, (fd.n, 1)))) * f1_p)
            lfdr = (np.transpose(np.tile(fd.pi0, (fd.n, 1)))) / f_p

            # Replacing those fdrs with 0 where the test statistic is so small
            # it only results in a value of nan in the p-value domain.
            lfdr[fd.p == 0] = 0
            print('completed!')
            if save_res:
                res = pd.DataFrame(
                    {"lfdr": [lfdr],
                     "f1_p": [f1_p],
                     "f_p": [f_p]})
            res.to_pickle(os.path.join(res_path, 'ground-truth.pkl'))
            del res
        pi0 = fd.pi0
        return lfdr, f_p, f1_p, pi0
    elif isinstance(fd, RadioSpatialFieldEstimated):
        try:
            print('\rRead in the ground truth lfdrs at sensors...', end="")
            loaded = pd.read_pickle(
                os.path.join(res_path, 'ground-truth-sen.pkl'))
            lfdr = loaded['lfdr'][0]
            f1_p = loaded['f1_p'][0]
            f_p = loaded['f_p'][0]
            print('completed!')

        except FileNotFoundError:
            print('\rCompute the ground truth lfdrs at sensors...', end="")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                f1_p = get_p_alt_pdf_jnt(fd, fd.p)

            f_p = (np.transpose(np.tile(fd.pi0, (fd.n, 1)))
                   + (1-np.transpose(np.tile(fd.pi0, (fd.n, 1)))) * f1_p)
            lfdr = (np.transpose(np.tile(fd.pi0, (fd.n, 1)))) / f_p

            # Replacing those fdrs with 0 where the test statistic is so small
            # it only results in a value of nan in the p-value domain.
            lfdr[fd.p == 0] = 0
            print('completed!')
            if save_res:
                res = pd.DataFrame(
                    {"lfdr": [lfdr],
                     "f1_p": [f1_p],
                     "f_p": [f_p]})
            res.to_pickle(os.path.join(res_path, 'ground-truth-sen.pkl'))
            del res
        pi0 = fd.pi0
        return lfdr, f_p, f1_p, pi0

def ipl_lfdrs(res_path, res_str, lfdrs_sen, sen_cds, fd_dim, num_gp,
              rbf='phs2', eps=1):
    """
    Interpolate lfdrs.

    Parameters
    ----------
    res_path: str
        The path to where the data is stored.
    res_str: str
        The name of the file where the current results are stored.
    lfdrs_sen : float array
        The lfdrs at the sensors
    sen_cds : int array
        The sensor coordinates
    fd_dim : tuple
        The size of the field in (y, x) direction
    num_gp : int
        The number of pixels in the field
    rbf : str, optional
        The type of radial basis function for interpolation.
        The default is 'phs2'.
    eps : float, optional
        The scaling parameter. For certain rbfs, like TPS, has no influence
        on the result UNLESS smooth is non-zero. Yet, modifying eps for these
        type of RBFs is not necessary, controlling smooth is enough. The
        default is 1.

    Returns
    -------
    lfdrs_int_and_smoo : array
        The interpolated and smoothed lfdrs.

    """
    try:
        res = pd.read_pickle(os.path.join(res_path, res_str + '.pkl'))
        lfdrs_int = res['lfdr_ipl'][0]
    except KeyError:
        nMC = lfdrs_sen.shape[0]
        gp_crds = np.squeeze(np.array([
            (np.remainder(np.arange(num_gp), fd_dim[0]),
            np.array((np.arange(num_gp)/fd_dim[0])).astype(int))])).transpose()
        lfdrs_int = np.zeros((nMC, num_gp))
        for mc in np.arange(0, nMC, 1):
            rbfi = RBFInterpolant(sen_cds[mc, :, :], lfdrs_sen[mc, :],
                                  phi=rbf, sigma=0)
            lfdrs_int[mc, :] = rbfi(gp_crds).reshape(np.prod(fd_dim))
        for mc in np.arange(lfdrs_sen.shape[0]):
            lfdrs_int[mc, :] = clip_lfdrs(lfdrs_int[mc, :])
        new_res = pd.concat([res, pd.DataFrame({"lfdr_ipl":[lfdrs_int]})],
                            axis=1)
        new_res.to_pickle(os.path.join(res_path, res_str + '.pkl'))
    return lfdrs_int

def is_d_ok(a_run, b_run):
    """Returns which multivariate entries can be used

    Parameters
    ----------
    a_run : numpy array
        The values of the first beta shape parameters to be checked for
        plausibility.
    b_run : _type_
        The values of the second beta shape parameters to be checked for
        plausibility.

    Returns
    -------
    boolean
        If the given values of the beta shape parameters are ok.
    """
    a_int_max = 10
    a_int_min = 0
    b_int_max = 10
    b_int_min = 0
    return (a_int_max >= a_run and
            a_int_min <= a_run and
            b_int_max >= b_run and
            b_int_min <= b_run)

def is_k_ok(w_hat):
    """Returns which components of the multivariate beta mixture model can be
    used.

    Parameters
    ----------
    w_hat : numpy array
        The weights for each mixture component.

    Returns
    -------
    numpy array
        A one-dimensional numpy array with indicators if a mixture component is
        save to use or not. If a weight of a mixture component is negative, it
        shall not be used.
    """
    use_k_idc = np.all([w_hat >= 0], axis=0)
    return use_k_idc

def sls_get_disvec(sen_cds, s):
    dis_vec = np.zeros(sen_cds.shape[0])
    for i in np.arange(dis_vec.size):
        dis_vec[i] = np.sqrt(
            (sen_cds[i, 0] - s[0]) ** 2 + (sen_cds[i, 1] - s[1]) ** 2)
    return dis_vec

def sns_get_disvec(dims, s):
    m = dims[0]*dims[1]
    dis_vec = np.zeros(m)
    for i in np.arange(dims[0]):
        dis_vec[(i*dims[1]):((i+1)*dims[1])] = np.sqrt(
            (i-s[0])**2+(np.arange(dims[1])-s[1])**2)
    return dis_vec

def sns_pis_2D_aux_func(p, dims, tau, h, kernel, exclude_sen=True):
    pv_vec = p.flatten()
    scr_idx = np.where(pv_vec >= tau)
    p_est = np.zeros(shape=dims)

    if not exclude_sen:
        for i in np.arange(dims[0]):
            for j in np.arange(dims[1]):
                s = np.array([i, j])
                dis_vec = sns_get_disvec(dims, s)
                kht = stats.norm.pdf(dis_vec, 0, h)
                if kernel == 'gauss':
                    kht = stats.norm.pdf(dis_vec, 0, h)
                elif kernel == 'tophat':
                    kht = 1 / h * (np.abs(dis_vec) < (h / 2))
                elif kernel == 'epa':
                    kht = 3/(4 * h) * np.max(
                        (1 - (dis_vec ** 2 / h ** 2), np.zeros(dis_vec.size)), 0)
                elif kernel == 'exp':
                    kht = np.exp(-np.abs(dis_vec) / h) * 1/h/2
                elif kernel == 'lin':
                    kht = 1 / h * (1 - np.abs(dis_vec/h)) * (np.abs(dis_vec) < h)
                elif kernel == 'cos':
                    kht = np.pi / (4 * h) * (
                        np.cos(np.pi / 2 * dis_vec / h) * (np.abs(dis_vec) < h))
                else:
                    print("Not a valid kernel!")
                    sys.exit()
                p_est[i, j] = np.min((
                    1-1e-5, np.sum(kht[scr_idx])/((1-tau)*np.sum(kht[pv_vec>0]))))
    else:
        for i in np.arange(dims[0]):
            for j in np.arange(dims[1]):
                s = np.array([i, j])
                dis_vec = sns_get_disvec(dims, s)
                if kernel == 'gauss':
                    kht = stats.norm.pdf(dis_vec, 0, h)
                elif kernel == 'tophat':
                    kht = 1 / h * (np.abs(dis_vec) < (h / 2))
                elif kernel == 'epa':
                    kht = 3/(4 * h) * np.max(
                        (1 - (dis_vec ** 2 / h ** 2), np.zeros(dis_vec.size)), 0)
                elif kernel == 'exp':
                    kht = np.exp(-np.abs(dis_vec) / h) * 1/h/2
                elif kernel == 'lin':
                    kht = 1 / h * (1 - np.abs(dis_vec/h)) * (np.abs(dis_vec) < h)
                elif kernel == 'cos':
                    kht = np.pi / (4 * h) * (
                        np.cos(np.pi / 2 * dis_vec / h) * (np.abs(dis_vec) < h))
                else:
                    print("Not a valid kernel!")
                    sys.exit()
                kht[np.where(dis_vec == 0)] = 0
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    p_est[i, j] = np.min((
                        1-1e-5, np.sum(kht[scr_idx])/(
                            (1-tau)*np.sum(kht[pv_vec>0]))))
    return 1-p_est.flatten()

def mbm_em(p, pi_k_0, a_k_0, eps):
    """Apply EM to estimate the parameters of a univariate MBM.

    Parameters
    ----------
    p : numpy array
        The p-values
    pi_k_0 : numpy array
        The initial values for the component mixiing weights.
    a_k_0 : numpy array
        The initial values for the beta distribution shape parameter.
    eps : float
        The convergence threshold for EM.
    """
    def a_mle(v, p):
        return np.max((0, - np.sum(v) / np.sum(v * np.log(p))))


    def pi_mle(v, N):
        return np.min((1, np.max((0, np.sum(v) / N))))
    K = np.size(pi_k_0)
    N = np.size(p)
    a_k = np.copy(a_k_0)
    pi_k = np.copy(pi_k_0)
    if K == 1:
        a_k = a_k[np.newaxis]
        pi_k = pi_k[np.newaxis]
    v_k = np.zeros((K, N))
    rel_diff = 1
    llf = mbm_llf(p, pi_k, a_k)

    a_k_iter = np.copy(a_k[:, np.newaxis])
    pi_k_iter = np.copy(pi_k[:, np.newaxis])
    v_k_iter = np.copy(v_k[:, :, np.newaxis])
    llf_iter = np.array(np.copy(llf))[np.newaxis]
    while rel_diff > eps:
        # E-Step
        v_k_denom = np.zeros(N)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for k in np.arange(K):
                v_k_denom = v_k_denom + pi_k[k] * a_k[k] * p ** (a_k[k] - 1)
            for k in np.arange(K):
                v_k[k, :] = pi_k[k] * a_k[k] * p ** (a_k[k] - 1) / v_k_denom
        v_k[np.where(np.isnan(v_k))] = 1
        v_k_iter = np.concatenate((v_k_iter, v_k[:, :, np.newaxis]), 2)
        # M-Step
        for k in np.arange(K):
            a_k[k] = a_mle(v_k[k, :], p)
            pi_k[k] = pi_mle(v_k[k, :],  N)
        pi_k_iter = np.concatenate((pi_k_iter, pi_k[:, np.newaxis]), 1)
        a_k_iter = np.concatenate((a_k_iter, a_k[:, np.newaxis]), 1)
        # Check for convergence
        llf_updated = mbm_llf(p, pi_k, a_k)
        diff = llf_updated - llf
        if np.size(llf_iter) == 1:
            rel_diff = 1
        else:
            rel_diff = np.sign(diff) * np.abs(diff/llf)
        llf = np.copy(llf_updated)
        llf_iter = np.concatenate((llf_iter, np.array(llf)[np.newaxis]))
    if rel_diff < 0:
        pi_k = pi_k_iter[:, -2]
        a_k = a_k_iter[:, -2]
        llf_iter = llf_iter[:-1]
    if llf_iter[-1] < llf_iter[0]:
        pi_k = pi_k_iter[:, 0]
        a_k = a_k_iter[:, 0]
        llf_iter = np.array(llf_iter[0])[np.newaxis]
    return pi_k, a_k, llf_iter

def mbm_llf(p, pi_k, a_k):
    """Compute the log likelihood function for a univariate MBM.

    Parameters
    ----------
    p : numpy array
        The data points for which the LLF is to be computed
    pi_k : numpy array
        The component mixing weights.
    a_k : numpy array
        The beta distribution shape parameters.

    Returns
    -------
    float
        The log likelihood function.
    """
    lf = np.zeros(np.size(p))
    K = np.size(pi_k)
    if K == 1:
        pi_k = pi_k[np.newaxis]
        a_k = a_k[np.newaxis]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for k in np.arange(K):
            lf = lf + (pi_k[k] * a_k[k] * p ** (a_k[k] - 1))
    llf = np.log(lf)
    # Wherever the llf is inf, replace it with the largest, non-inf number
    # occuring.
    try:
        llf[np.where(llf == np.inf)] = np.max(llf[llf < np.inf])
    except ValueError:
        print("Stuck here!")
    # Limiting to the same range as smom!
    p_lim = 1e-10# 0 #1e-10
    zero_idx = np.reshape(p < p_lim, llf.shape)
    llf[zero_idx] = 0
    return np.sum((llf))

def parallel_smom(fd, par_type, dat_path, max_wrk, partition, quant_bits,
                  sensoring_thr):
    """Estimate the alternative p-value PDF for lfdr-sMoM with parallelization.

    Parameters
    ----------
    fd : RadioSpatialField/RadioSpatialFieldEstimated
        The field the lfdrs are to be estimated for.
    par_type : str
        The smom parametrization type. Recommended: 'stan' for standard.
    dat_path : str
        The path to where the data is stored.
    partition : str
        The type of partition for obtaining the p-value vectors.
    quant_bits : int or None
        The number of bits used for quantization.
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring occurs.

    Returns
    -------
    list
        The list with all estimated quantities.
    """
    with open(os.path.join(dat_path, "..", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        if partition == 'spatial':
            mom_d = mom_d[
                np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
        del ld_par

    p_val = fd.p
    num_wrk = np.min((max_wrk, os.cpu_count() - 1))
    print(f"Starting a parallel pool with {num_wrk} workers.")
    par_pl = mp.Pool(num_wrk)
    p_val = fd.p
    if partition == 'spatial':
        rtns = par_pl.map(partial(
            single_run_smom_spatial, dat_path=dat_path, par_type=par_type,
            quant_bits=quant_bits, sensoring_thr=sensoring_thr), p_val)
    elif partition == 'random':
        rtns = par_pl.map(partial(
            single_run_smom_random, dat_path=dat_path, par_type=par_type,
            quant_bits=quant_bits, sensoring_thr=sensoring_thr), p_val)
    par_pl.close()

    # Setting up null proportion and alternative density
    pi_0_hat = np.zeros((fd.n_MC))
    f1_hat = np.zeros((fd.n_MC, fd.n))

    # Setting up the variables for storing the results
    mom_lam_hats = np.zeros((fd.n_MC, np.max(mom_k)))
    mom_a_hats = np.zeros((fd.n_MC, np.max(mom_k), np.max(mom_d)))
    mom_p_pdf = np.zeros((fd.n_MC, fd.n))
    mom_p_cdf = np.zeros((fd.n_MC, fd.n))
    mom_sel_k = np.zeros((fd.n_MC))
    mom_sel_d = np.zeros((fd.n_MC))
    ex_time = np.zeros(fd.n_MC)
    mom_diff_best = np.zeros(fd.n_MC)
    for mc in np.arange(0, fd.n_MC, 1):
        # Unpacking the results of the spectral method of moments
        mom_a_hats[mc, :, :] = rtns[mc][0]
        mom_lam_hats[mc, :] = rtns[mc][1]
        mom_p_pdf[mc, :] = rtns[mc][2]
        mom_p_cdf[mc, :] = rtns[mc][3]
        mom_diff_best[mc] = rtns[mc][4]
        mom_sel_k[mc] = rtns[mc][5]
        mom_sel_d[mc] = rtns[mc][6]
        ex_time[mc] = rtns[mc][7]
        # Estimating the alternative density and null proportion by Pounds
        # method
        (pi_0_hat[mc], f1_hat[mc, :]) = apply_pounds_estimator(
            [mom_p_pdf[mc, :], mom_a_hats[mc, :, :],
             np.ones(mom_a_hats[mc, :, :].shape), mom_lam_hats[mc, :],
             mom_sel_d[mc], mom_sel_k[mc]], typ='mbm')
    return (f1_hat, pi_0_hat, mom_lam_hats, mom_a_hats,
            mom_p_pdf, mom_p_cdf, mom_diff_best, mom_sel_k, mom_sel_d,
            ex_time)

def single_run_sls(lfdr, sen_cds, h, kernel):
    """
    TODO: UPdate comment.

    Estimate prior alternative probabilities for Cai2021's LAWS procedure.

    For the sampled spatial field.

    Parameters
    ----------
    p_val_vec : numpy array
        The p-value vector for a single MC run.
    sen_cds: numpy array
        The MC x x-coordinate x y-coordinate matrix of sensor locations.
    h : float
        TODO: Confirm: A bandwidth parameter? for the smoothing?

    Returns
    -------
    numpy array
        The estimated prior alternative probabilities.

    """
    # Choose the right kernel
    if kernel == 'gauss':
        def ker_fct(h, dis):
            return (1 / np.sqrt(2 * np.pi * h ** 2)
                    * np.exp(-1 / 2 / h ** 2 * dis ** 2))
    elif kernel == 'tophat':
        def ker_fct(h, dis):
            return 1 / h * (np.abs(dis) < (h / 2))
    elif kernel == 'epa':
        def ker_fct(h, dis):
            return 3/(4 * h) * np.max(
                (1 - (dis ** 2 / h ** 2), np.zeros(dis.size)), 0)
    elif kernel == 'exp':
        def ker_fct(h, dis):
            return np.exp(-np.abs(dis) / h) * 1/h/2
    elif kernel == 'lin':
        def ker_fct(h, dis):
            return 1 / h * (1 - np.abs(dis/h)) * (np.abs(dis) < h)
    elif kernel == 'cos':
        def ker_fct(h, dis):
            return np.pi / (4 * h) * (
                np.cos(np.pi / 2 * dis / h) * (np.abs(dis) < h))
    else:
        print("Not a valid kernel!")
        sys.exit()

    pi1_est = np.zeros(lfdr.shape)
    for i in np.arange(sen_cds.shape[0]):
        s = np.array([sen_cds[i, 0], sen_cds[i, 1]])
        dis_vec = sls_get_disvec(sen_cds, s)
        dis_vec[dis_vec == 0] = 1e10

        # pi1_est[i] = 1 - np.sum(lfdr_vec * kht)/np.sum(kht)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            pi1_est[i] = 1 - (np.sum(ker_fct(h, dis_vec) * lfdr)
                              / np.sum(ker_fct(h, dis_vec)))
    return pi1_est.flatten()

def single_run_sns(p_val_vec, sen_idx_vec, siz, laws_sthr, h, kernel,
                    exclude_sen):
    p_val = np.zeros(np.prod(siz))
    p_val[sen_idx_vec] = p_val_vec
    p_val = p_val.reshape(siz)

    _, bh_th = bh(p_val_vec[np.newaxis, :], laws_sthr, get_pval_thr_num=True)
    pis_hat = sns_pis_2D_aux_func(p_val, siz,
                                   tau=bh_th[0], h=h, kernel=kernel,
                                   exclude_sen=exclude_sen)
    return pis_hat

def single_run_smom_em(dat, a0, pi0, cvg_thr):
        use_a_0 = np.all(
            (a0 > 0, (np.tile(pi0[:, np.newaxis], (1, np.size(a0, 1))) > 0)),
            0)
        pi0_active = np.tile(pi0[:, np.newaxis], (1, np.size(a0, 1)))[use_a_0]
        pi0_active = pi0_active / np.sum(pi0_active)
        a0_active = a0[use_a_0]
        mod_ord_sel = np.size(pi0_active)
        start_time = time.time()

        (pi_k, a_k, llr_iter) = mbm_em(dat, pi0_active, a0_active, cvg_thr)
        num_it = np.size(llr_iter) - 1

        # PDF
        f_p_k = np.zeros((mod_ord_sel, np.size(dat)))
        f_p = np.zeros(np.size(dat))

        # Segmentation fault may occur for super small p-values below ~1e-305.
        dat[np.where(dat<1e-305)] = 1e-305
        for k in np.arange(mod_ord_sel):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                f_p_k[k, :] = stats.beta.pdf(dat, a_k[k], 1)
            f_p = f_p + pi_k[k] * f_p_k[k, :]

        end_time = time.time()
        ex_time = end_time - start_time

        return f_p, mod_ord_sel, pi_k, a_k, ex_time, num_it

def single_run_smom_random(p_val, dat_path, par_type, quant_bits,
                           sensoring_thr):
    """Apply smom with random partitioning for a single MC run.

    Parameters
    ----------
    p_val : numpy array
        The p-values for this MC run.
    dat_path : str
        The path to where the data is stored.
    par_type : str
        The smom parametrization. Recommended: 'stan' for standard
    quant_bits : int or None
        The number of bits for quantization. If None, then no quantization
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring.

    Returns
    -------
    list
        The quantities estimated by sMoM.
    """
    # Reading in method of moments parameters
    with open(os.path.join(dat_path, "..", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        del ld_par
    start_time = time.time()

    # Setting up the grid for the goodness-of-fit statistic
    if dis_msr == 'js' or dis_msr == 'kl':
        bin_wdt = 10/p_val.size  # Need to make sure that there are few empty
        # bins.
    elif dis_msr == 'ks' or dis_msr == 'was':
        bin_wdt = 3/p_val.size  # Need as many bins as possible for EDF-based
        # measures
        bin_wdt = 1/1000
    else:
        print("Invalid distance measure")
    grd = np.arange(1e-10, 1, bin_wdt)  # the grid
    grd_cut_off_idx = -1
    bns = np.arange(1e-10-bin_wdt/2, 1+bin_wdt/2, bin_wdt)  # the bin edges

    if quant_bits is not None:
        with open(os.path.join(
            dat_path, '..',
            f'quan_{quant_bits}Bit_sensoring_at_{sensoring_thr}') + '_par.pkl',
            'rb') as input:
            loaded = pickle.load(input)
            (bns, bin_wdt, grd) = loaded[0], loaded[1], loaded[2]

    # Spatial division of the data in tiles
    N = p_val.size

    # Specification of the tile parameters
    num_tiles = np.floor(N/mom_d).astype(int)
    N_tilde = num_tiles*mom_d.astype(int)

    # Computation of the edf on the grid for g.o.f
    grd_emp_prob = np.histogram(p_val, bins=bns, density=True)[0]*bin_wdt
    grd_edf = np.cumsum(grd_emp_prob)

    diff_best = np.inf
    sel_d = np.nan
    sel_k = np.nan
    if dis_msr == 'ks':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.max(np.abs((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx]), 1)
    elif dis_msr == 'msd':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.mean(((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx])**2, 1)
    elif dis_msr == 'was':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = wasserstein_distance(
                    grd_edf, grd_cdf[t_idx, :])
            return diff_k
    elif dis_msr == 'kl':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = entropy(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)
            return diff_k
    elif dis_msr == 'js':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = jensenshannon(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)**2
            return diff_k
    for d_idx in np.arange(0, mom_d.size, 1):
        # Setting up the p-values divided into tiles
        if (not np.isnan(sel_d)) and sel_d < mom_d[d_idx-1]:
            break
        for tr_idx in np.arange(0, mom_n_tr, 1):
            shuffled_pval_idx = np.random.permutation(
                np.arange(N))[0:N_tilde[d_idx]].reshape(
                    (num_tiles[d_idx], mom_d[d_idx]))
            p_div = p_val[shuffled_pval_idx]
            p_div[np.where(p_div == 0)] = np.min(p_div[np.nonzero(p_div)])

            # Parameter estimation by spectral method of moments
            diff_all_k = np.zeros(mom_k.size)
            for (k_idx, k) in enumerate(mom_k):
                if k < mom_d[d_idx]:  # Size of multivariate vectors limits
                    # the number of mixture components.
                    a_hat, b_hat, w_hat, grd_pdf, grd_cdf = (
                        smom_functions.learnMBM(p_div, num_tiles[d_idx],
                        mom_d[d_idx], k, grd, mom_reps_eta, gaussian_eta=True))
                    # Goodness of fit
                    diff_k = dis_msr_fct(grd_pdf, grd_cdf)
                    try:
                        min_idx = np.nanargmin(diff_k)
                        diff_all_k[k_idx] = diff_k[min_idx]
                        if (diff_all_k[k_idx]
                            - diff_all_k[np.max((k_idx - 1, 0))] <= 0):
                            if diff_all_k[k_idx] - diff_best <= 0:
                                a_hat_win = np.copy(a_hat[min_idx, :, :])
                                b_hat_win = np.copy(b_hat[min_idx, :, :])
                                w_hat_win = np.copy(w_hat[min_idx, :])
                                diff_best = diff_all_k[k_idx]
                                sel_k = k
                                sel_d = mom_d[d_idx]
                        else:
                            break
                    except ValueError:
                        # print(['No valid result for this parametrization'
                        #       ' with averaging!'])
                        failed_at_least_once = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # PDF marginalized over coordinates
        beta_pdf_hat_av = get_pdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
        # CDF marginalized over coordinates
        beta_cdf_hat_av = get_cdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
    # Scaling the weights back to sum to 1
    w_hat_win[np.where(w_hat_win < 0)[0]] = 0
    w_hat_win = w_hat_win/np.sum(w_hat_win)
    # print('\rCompleted run {mc}/{nMC}'.format(mc=mc+1, nMC=mc), end="")
    a_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    b_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    w_hat_rtd = np.zeros((np.max(mom_k)))
    a_hat_rtd[0:sel_k, 0:sel_d] = a_hat_win
    b_hat_rtd[0:sel_k, 0:sel_d] = b_hat_win
    w_hat_rtd[0:sel_k] = w_hat_win
    ex_time = time.time() - start_time
    return (a_hat_rtd, w_hat_rtd, beta_pdf_hat_av,
            beta_cdf_hat_av, diff_best, sel_k, sel_d, ex_time)

def single_run_smom_spatial(p_val, dat_path, par_type, quant_bits,
                            sensoring_thr):
    """Apply smom with spatial partitioning for a single MC run.

    Parameters
    ----------
    p_val : numpy array
        The p-values for this MC run.
    dat_path : str
        The path to where the data is stored.
    par_type : str
        The smom parametrization. Recommended: 'stan' for standard
    quant_bits : int or None
        The number of bits for quantization. If None, then no quantization
    sensoring_thr : float
        The sensoring threshold. If = 1, then no sensoring.

    Returns
    -------
    list
        The quantities estimated by sMoM.
    """
    with open(os.path.join(dat_path, "..", "smom_par_")
        + par_type + '.pkl', 'rb') as input:
        ld_par = pickle.load(input)
        [mom_k, mom_d, mom_n_tr, mom_reps_eta, dis_msr] = [x for x in ld_par]
        del ld_par
    # Reading in method of moments parameters
    mom_d = mom_d[np.where(np.sqrt(mom_d) - np.sqrt(mom_d).astype(int)==0)]
    start_time = time.time()
    # Setting up the grid for the goodness-of-fit statistic
    if dis_msr == 'js' or dis_msr == 'kl':
        bin_wdt = 10/p_val.size  # Need to make sure that there are few empty
        # bins.
    elif dis_msr == 'ks' or dis_msr == 'was':
        bin_wdt = 3/p_val.size  # Need as many bins as possible for EDF-based
        # measures
        bin_wdt = 1/1000
    else:
        print("Invalid distance measure")

    grd = np.arange(1e-10, 1, bin_wdt)  # the grid
    grd_cut_off_idx = -1
    bns = np.arange(1e-10-bin_wdt/2, 1+bin_wdt/2, bin_wdt)  # the bin edges

    if quant_bits is not None:
        with open(os.path.join(dat_path, '..',
            f'quan_{quant_bits}Bit_sensoring_at_{sensoring_thr}') + '_par.pkl',
            'rb') as input:
            loaded = pickle.load(input)
            (bins, bin_wdt, grd) = loaded[0], loaded[1], loaded[2]


    # Spatial division of the data in tiles
    N = p_val.size
    dim = (int(np.sqrt(N)), int(np.sqrt(N)))

    # Number of tiles along x and y axis
    tls_per_x = (np.floor(np.sqrt(N/mom_d))).astype(int)
    tls_per_y = (np.floor(np.sqrt(N/mom_d))).astype(int)

    # Specification of the tile parameters
    num_tiles = tls_per_x * tls_per_y
    tls_x_len = (np.floor(dim[0]/tls_per_x)).astype(int)
    tls_y_len = (np.floor(dim[1]/tls_per_y)).astype(int)

    # Computation of the edf on the grid for g.o.f
    grd_emp_prob = np.histogram(p_val, bins=bns, density=True)[0]*bin_wdt
    grd_edf = np.cumsum(grd_emp_prob)
    # Initialization of g.o.f measures
    diff_best = np.inf
    sel_d = np.nan
    sel_k = np.nan

    if dis_msr == 'ks':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.max(np.abs((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx]), 1)
    elif dis_msr == 'msd':
        def dis_msr_fct(grd_pdf, grd_cdf):
            return np.mean(((grd_edf-grd_cdf)[:, 0:grd_cut_off_idx])**2, 1)
    elif dis_msr == 'was':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = wasserstein_distance(
                    grd_edf, grd_cdf[t_idx, :])
            return diff_k
    elif dis_msr == 'kl':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = entropy(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)
            return diff_k
    elif dis_msr == 'js':
        def dis_msr_fct(grd_pdf, grd_cdf):
            diff_k = np.zeros(mom_reps_eta)
            for t_idx in np.arange(mom_reps_eta):
                diff_k[t_idx] = jensenshannon(
                    grd_emp_prob[1:], grd_pdf[t_idx, 1:]*bin_wdt)**2
            return diff_k

    for d_idx in np.arange(0, mom_d.size, 1):
        if (not np.isnan(sel_d)) and sel_d < mom_d[d_idx-1]:
            break
        # Setting up the p-values divided into tiles
        p_div = np.zeros((
            num_tiles[d_idx], tls_x_len[d_idx]*tls_y_len[d_idx])) + np.nan
        for tr_idx in np.arange(0, mom_n_tr, 1):
            for spa_div_x_idx in np.arange(0, tls_per_x[d_idx], 1):
                for spa_div_y_idx in np.arange(0, tls_per_y[d_idx], 1):
                    tls_sel_mat = np.zeros(dim, dtype=bool)
                    (tle_starts_at_x_idx, tle_starts_at_y_idx) = (
                        spa_div_x_idx*tls_x_len[d_idx], spa_div_y_idx
                        * tls_y_len[d_idx])
                    tls_sel_mat[
                        tle_starts_at_x_idx:tls_x_len[d_idx]
                        +tle_starts_at_x_idx,
                        tle_starts_at_y_idx:tle_starts_at_y_idx+tls_y_len[
                            d_idx]] = True
                    p_vec = np.random.choice(
                        p_val, size=np.prod(dim),
                        replace=False).reshape(dim)[tls_sel_mat]
                    # Uncomment this to exclude the pile at 0 from the fitting
                    if ~np.all(p_vec == 0):
                        p_vec[np.where(p_vec == 0)] = np.min(
                            p_vec[np.where(p_vec != 0)])
                        p_div[spa_div_x_idx*tls_per_x[d_idx]
                        + spa_div_y_idx, :] = (
                            np.random.permutation(p_vec))
                    # p_div[spa_div_x_idx*tls_per_x + spa_div_y_idx, :] = (
                    #     np.random.permutation(p_vec))
            # Uncomment to exclude tiles with only 0 from the fitting
            tiles_for_fitting = np.where(~np.isnan(np.sum(p_div, 1)))[0]
            # Uncomment to include the pile at0
            # tiles_for_fitting = np.arange(0, num_tiles, 1)

            # Paramter estimation by spectral method of moments
            diff_all_k = np.zeros(mom_k.size)

            for (k_idx, k) in enumerate(mom_k):
                if k < mom_d[d_idx]:  # Size of multivariate vectors limits
                    # the number of mixture components.
                    a_hat, b_hat, w_hat, grd_pdf, grd_cdf = (
                        smom_functions.learnMBM(
                        p_div[tiles_for_fitting, :], tiles_for_fitting.size,
                        mom_d[d_idx], k, grd, mom_reps_eta,
                        gaussian_eta=True))

                    # Goodness of fit
                    diff_k = dis_msr_fct(grd_pdf, grd_cdf)
                    try:
                        min_idx = np.nanargmin(diff_k)
                        diff_all_k[k_idx] = diff_k[min_idx]
                        if diff_all_k[k_idx] - diff_all_k[
                            np.max((k_idx - 1, 0))] <= 0:
                            if diff_all_k[k_idx] - diff_best <= 0:
                                a_hat_win = np.copy(a_hat[min_idx, :, :])
                                b_hat_win = np.copy(b_hat[min_idx, :, :])
                                w_hat_win = np.copy(w_hat[min_idx, :])
                                diff_best = diff_all_k[k_idx]
                                sel_k = k
                                sel_d = mom_d[d_idx]
                        else:
                            break
                    except ValueError:
                        # print(['No valid result for this parametrization'
                        #       ' with averaging!'])
                        failed_at_least_once = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # PDF marginalized over coordinates
        beta_pdf_hat_av = get_pdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
        # CDF marginalized over coordinates
        beta_cdf_hat_av = get_cdf_multivariate_mbm(
            p_val, a_hat_win, b_hat_win, w_hat_win, sel_d, sel_k)
    # Scaling the weights back to sum to 1
    w_hat_win[np.where(w_hat_win < 0)[0]] = 0
    w_hat_win = w_hat_win/np.sum(w_hat_win)
    # print('\rCompleted run {mc}/{nMC}'.format(mc=mc+1, nMC=mc), end="")
    a_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    b_hat_rtd = np.zeros((np.max(mom_k), np.max(mom_d)))
    w_hat_rtd = np.zeros((np.max(mom_k)))
    a_hat_rtd[0:sel_k, 0:sel_d] = a_hat_win
    b_hat_rtd[0:sel_k, 0:sel_d] = b_hat_win
    w_hat_rtd[0:sel_k] = w_hat_win
    ex_time = time.time() - start_time
    return (a_hat_rtd, w_hat_rtd, beta_pdf_hat_av,
            beta_cdf_hat_av, diff_best, sel_k, sel_d, ex_time)

def single_run_mbm_em(p, mod_ords, n_reps, cvg_thr):
    """Apply MBM-EM wfor a single MC run.

    Parameters
    ----------
    p : numpy array
        The p-values for this MC run.
    mod_ords : numpy array
        The candidate model orders
    n_reps : int
        The number of random initializations for each model order.
    cvg_thr: float
        The EM convergence threshold
    Returns
    -------
    list
        The results of EM.
    """
    a_cnds = []
    pi_cnds = []
    num_iter_cnds = np.zeros(np.size(mod_ords))
    bic_vls = np.zeros(np.size(mod_ords)) + np.inf
    start_time = time.time()

    for (cnd_idx, K_cnd) in enumerate(mod_ords):
        if cnd_idx > 2:
            # If there was no improvement in the BIC in the last two candidates
            # don't continue with higher orders to save computation times.
            if(((bic_vls[cnd_idx-2] - bic_vls[cnd_idx-3]) > 0)
               and ((bic_vls[cnd_idx-1] - bic_vls[cnd_idx-2]) > 0)):
                break
        # Quantities for this candidate, mbm
        pi_cnd = np.zeros((K_cnd, n_reps))
        a_cnd = np.zeros((K_cnd, n_reps))
        llr_cnd = np.zeros(n_reps)
        num_iter_cnd = np.zeros(n_reps)
        def sgl_ran_run(run_idx):
            # Initial values for this run, mbm
            pi_run = stats.uniform.rvs(size=K_cnd)
            pi_run = np.array(pi_run/np.sum(pi_run))
            a_run = np.array(2*stats.uniform.rvs(size=K_cnd))
            # Application of EM, mbm
            (pi_cnd, a_cnd, llr_iter) = (
                mbm_em(p, pi_run, a_run, cvg_thr))
            llr_cnd = llr_iter[-1]
            num_iter = np.size(llr_iter) - 1
            return (a_cnd, pi_cnd, llr_cnd, num_iter)

        for run_idx in np.arange(n_reps):
            (a_cnd[:, run_idx], pi_cnd[:, run_idx],
              llr_cnd[run_idx], num_iter_cnd[run_idx]) = sgl_ran_run(run_idx)

        def bic(p, pi_k, a_k, fix0=False):
            K = np.size(pi_k)
            if fix0:
                dof = 2*K - 2
            else:
                dof = 2*K-1
            p_lim = 1e-10# 0 #1e-10
            N = (np.sum(p > p_lim))
            return dof*np.log(N) - 2 * (mbm_llf(p, pi_k, a_k))
        # Find best run for this candidate model order, mbm
        a_cnds.append(a_cnd[:, np.argmax(llr_cnd)])
        pi_cnds.append(pi_cnd[:, np.argmax(llr_cnd)])
        bic_vls[cnd_idx] = bic(
            p, pi_cnds[cnd_idx], a_cnds[cnd_idx])
        num_iter_cnds[cnd_idx] = num_iter_cnd[np.argmax(llr_cnd)]

    # Model order selection
    mod_ord_sel = mod_ords[np.argmin(bic_vls)]
    # print('Selected {k} components with mbm'.format(k=K_sel_mbm))
    a_k = a_cnds[np.argmin(bic_vls)]
    pi_k = pi_cnds[np.argmin(bic_vls)]
    num_iter = num_iter_cnds[np.argmin(bic_vls)]

    end_time = time.time()
    ex_time = end_time - start_time
    return (mod_ord_sel, a_k, pi_k, num_iter, bic_vls, ex_time)
