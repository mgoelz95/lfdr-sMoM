#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file provides detection capabilities.

@author: Martin Gölz
"""
import numpy as np

class DetectionResult(object):
    """
    The objects of this class represent the result of applying a multiple
    hypothesis testing procedure.

    @author: Martin Goelz
    """

    def __init__(self, nam, tar, thr, r_tru, r_det, cpl=True,
                 mis_mat=None, sen=False):
        """
        Store a detection results and compute the performance measures.

        Parameters
        ----------
        nam : string
            The identifier of this detection result.
        tar : string
            The performance quantity targetted to control.
        thr : float
            The nominal level of the targetted performance quantity.
        r_tru : array
            The true discovery pattern.
        r_det : array
            The detected discovery pattern
        sen : boolean, optional
            Indicator for whether these results were obtained for sensors.
            The default is False.

        Returns
        -------
        None.

        @author: Martin Goelz
        """

        if np.array(r_tru).shape == np.array(r_det).shape:
            if cpl==True:
                mis_mat = np.zeros(r_det.shape)
            self.r_tru = r_tru
            self.r_det = r_det
            # Taking data points that have not been interpolated out of the
            # evaluation
            self.r_tru[mis_mat == 1] = 0
            self.r_det[mis_mat == 1] = 0
            self.nam = nam
            self.tar = tar
            self.thr = thr
            self.sen = sen

            #Performance measures: Single run quantities
            self.v = np.zeros(r_tru.shape).astype(int) #False Discoveries
            self.s = np.zeros(r_tru.shape).astype(int) #Correct Discoveries
            self.t = np.zeros(r_tru.shape).astype(int) #Missed Discoveries
            self.u = np.zeros(r_tru.shape).astype(int) #Correct Non-Discoveries
            self.v[self.r_tru == 0] = self.r_det[self.r_tru == 0] == 1
            self.s[self.r_tru == 1] = self.r_det[self.r_tru == 1] == 1
            self.t[self.r_tru == 1] = self.r_det[self.r_tru == 1] == 0
            self.u[self.r_tru == 0] = self.r_det[self.r_tru == 0] == 0
            self.fdp = np.sum(self.v, axis=1)/(
                np.sum(self.r_det, axis=1)+(
                np.sum(self.r_det, axis=1)== 0)*1)
            self.tdp = np.sum(self.s, axis=1)/(np.sum(self.r_tru,
                                                      axis=1)+
                                      (np.sum(self.r_tru,
                                              axis=1) == 0)*1)
            #Perfomance measures: Averages
            self.fdr = np.mean(self.fdp, axis=0)
            self.pow = np.mean(self.tdp, axis=0)
            self.fwer = np.sum(np.sum(self.v, axis=1)>0)/self.v.shape[0]
        else:
             print('Results not added: Dimensions of shape are not identical!')

    def __str__(self):
        out = ''
        if self.tar == 'fwer':
            fwer_tar = '(Target = {tar:.5f})'.format(tar=self.thr)
            fdr_tar = ''
        elif self.tar == 'fdr':
            fwer_tar = ''
            fdr_tar = '(Target = {tar:.5f})'.format(tar=self.thr)
        else:
            fwer_tar = ''
            fdr_tar = ''
        out += ('Procedure {pro}: \t \t FDR =\t{fdr:.4f}\t{fdr_tar}\n\
Procedure {pro}: \t \t\
FWER =\t{fwer:.4f}\t{fwer_tar}\n\
Procedure {pro}: \t \t\
power =\t{pwr:.4f}\t\n'.format(pro=self.nam, fdr=self.fdr,
                                         fwer=self.fwer, fdr_tar=fdr_tar,
                                         fwer_tar=fwer_tar, pwr=self.pow))
        return out

    @property
    def tar(self):
        """
        The performance measure to be controlled.

        Returns
        -------
        None.

        @author: Martin Goelz
        """

        return self._tar

    @tar.setter
    def tar(self, val):
        if val in('fdr', 'fwer'):
            self._tar = val
        else:
            self._tar = ''

    @tar.deleter
    def tar(self):
        del self._tar

# %% functions
def apply_dBH(p, r_tru, alp_vec, bit_budget=None, name='dBH'):
    """Obtain detection results with distributed BH from [Ermis2010].

    Parameters
    ----------
    p : numpy array
        An nMC x number of tests numpy array with p-values.
    r_tru : numpy array
        An nMC x number of tests numpy array of 0 and 1 indicating true H0 & H1
    alp_vec : numpy array
        Vector of nominal FDR levels.
    bit_budget : int, optional
        The number of maximum hops. The default is None, which results in all
        sensors transmitting.
    name : str, optional
        The name of the applied procedure, by default 'dBH'.

    Returns
    -------
    DetectionResult
        The detection results.
    """
    if bit_budget is None:
        bit_budget = p.shape[1]
    det_res = []
    for idx in np.arange(0, alp_vec.size, 1):
        det_res.append(DetectionResult(name, 'fdr', alp_vec[idx],
            r_tru, dis_bh(p, alp_vec[idx], bit_budget)[0]))
    return det_res

def apply_lfdr_detection(lfdrs, r_tru, alp_vec, name, sen):
    """Obtain detection results with lfdrs.

    Parameters
    ----------
    lfdrs : numpy array
        An nMC x number of tests numpy array with lfdrs.
    r_tru : numpy array
        An nMC x number of tests numpy array of 0 and 1 indicating true H0 & H1
    alp_vec : numpy array
        Vector of nominal FDR levels.
    name : str
        The name of the detection procedure.
    sen : boolean
        Indicating if sensors only

    Returns
    -------
    DetectionResult
        The detection results.
    """
    det_res = []
    for idx in np.arange(0, alp_vec.size, 1):
        det_res.append(DetectionResult('lfdrs ' + name, 'fdr', alp_vec[idx],
        r_tru, bh_loc_bayes(lfdrs, alp_vec[idx]), sen=sen))
    return det_res

# %%Procedures for FDR control
def a_bh(p, alp, lam=0.5):
    """
    Perform the adaptive Benjamini-Hochberg procedure [Storey 2002]. Estimates
    the proportion of true nulls among the tested hypotheses by assessing the
    level of the assumed uniform p-value distribution above a certain lam
    threshold. Assumes independent p-values under H_0.

    Parameters
    ----------
    p : array
        The MC x N matrix of p-values.
            - MC:   # of MC runs
            - N:    # of tested hypotheses
    alp : float
        The FWER nominal level.
    lam : float
        The threshold for estimating the null proportion. Standard value =.5

    Returns
    -------
    r : array
        The MC x N indicator matrix with an element = 1 if the corresponding
        hypothesis is rejected.
    k : int
        The index until which the sorted p-values were rejected.

    @Martin Goelz
    """

    sor_p = np.sort(p) #Sorting p-values in ascending order
    m = np.ma.size(p, axis=-1) #Retrieving the number of hypotheses
    MC = np.ma.size(p, axis=0) #Retrieving the number of MC runs

    #Estimating the proportion of nulls: compute inverse for efficiency
    inv_pi_0_hat = (1-lam)*m/(np.sum(p>(lam+np.zeros((MC, 1))), axis=1) + 1)

    #Identifying the cut-off index    
    k = np.zeros(MC, dtype=int)
    for mc in np.arange(MC):
        k[mc]  = np.max(
            np.where(
                sor_p[mc, :] <= inv_pi_0_hat[mc] * np.arange(1, m+1, 1)/m*alp),
            initial=-1)
    thr = sor_p[np.arange(0, MC), k].reshape(MC, 1)#Rejection thresholds
    thr[k == -1] = 0 #Those MC trials where no p-val is to be rejected
    r = p <= thr #Hypothesis testing

    return r, k

def bh(p, alp, mis_ipl=False, get_pval_thr_num=False):
    """
    Perform the standard Benjamini-Hochberg procedure.

    Parameters
    ----------
    p : array
        The MC x N matrix of p-values.
            - MC:   # of MC runs
            - N:    # of tested hypotheses
    alp : float
        The FWER nominal level.

    Returns
    -------
    r : array
        The MC x N indicator matrix with an element = 1 if the corresponding
        hypothesis is rejected.
    k : int
        The index until which the sorted p-values were rejected.
    mis_ipl: boolean
        An indicator whether the given values might miss some interpolation
        values. Needed for DT.

    @Martin Goelz
    """

    sor_p = np.sort(p) #Sorting p-values in ascending order
    m = np.ma.size(p, axis=-1) #Retrieving the number of hypotheses
    MC = np.ma.size(p, axis=0) #Retrieving the number of MC runs

    if mis_ipl==False:
        #Identifying the cut-off index
        k = np.zeros(MC, dtype=int)
        for mc in np.arange(MC):
            k[mc]  = np.max(
                np.where(sor_p[mc, :] <= np.arange(1, m+1, 1)/m*alp),
                initial=-1)
        thr = sor_p[np.arange(0, MC), k].reshape(MC, 1)#Rejection thresholds
        thr[k == -1] = 0 #Those MC trials where no p-val is to be rejected
        r = p <= thr #Hypothesis testing
    else:
        print("THE THRESHOLD MUST BE UPDATED FOR THIS BH PROCEDURE!")
        k = np.zeros(p.shape[0])
        thr = np.zeros(p.shape[0])
        r = np.zeros(p.shape)
        for mc in np.arange(0, p.shape[0], 1):
            m = np.sum(~np.isnan(sor_p[mc,:]))
            k[mc] = (np.argmax(sor_p[mc, np.where(~np.isnan(sor_p[mc,:]))] >
                              np.arange(1, m+1, 1)/m*alp+np.zeros((MC, 1)))- 1)
            thr[mc] = sor_p[mc, k[mc].astype(int)]#Rejection threshold
        thr[k == -1] = 0 #Those MC trials where no p-val is to be rejected
        for mc in np.arange(0, p.shape[0], 1):
            r[mc, np.where(~np.isnan(p[mc, :]))[0]] = p[
                mc,np.where(~np.isnan(p[mc, :]))[0]] <= thr[mc]
    if get_pval_thr_num:
        return r, thr
    else:
        return r, k

def bh_bayes(pos_nul, alp, z, mis_ipl=False):
    """
    Perform the standard Benjamini-Hochberg procedure for given empirical
    posterior null probabilities.

    Parameters
    ----------
    pos_nul : array
        The MC x N matrix of posterior null probabilities.
            - MC:   # of MC runs
            - N:    # of tested hypotheses
    alp : float
        The FWER nominal level.
    z : array
        The z-scores

    Returns
    -------
    r : array
        The MC x N indicator matrix with an element = 1 if the corresponding
        hypothesis is rejected.

    @Martin Goelz
    """

    cds_idx = pos_nul <= alp
    max_per_run = np.zeros(pos_nul.shape[0])
    for mc in np.arange(0, pos_nul.shape[0], 1):
        max_per_run[mc] = np.max(
            z[mc, cds_idx[mc, :]],
            initial=np.min(z[mc, np.where(~np.isnan(z[mc, :]))])-1)
    return z <= np.repeat(max_per_run[:, np.newaxis], pos_nul.shape[1], axis=1)


def bh_loc_bayes(lfdr, alp):
    """
    Perform the local fdr procedure.

    Finds the largest region for which the Bayesian FDR (summed and normalized
    local fdrs) is smaller than the nominal level.

    Parameters
    ----------
    lfdr : array
        The MC x N matrix of local fdrs.
            - MC:   # of MC runs
            - N:    # of tested hypotheses
    alp : float
        The FDR nominal level.

    Returns
    -------
    array
        The MC x N indicator matrix with an element = 1 if the corresponding
        hypothesis is rejected.

    @Martin Goelz
    """
    srt_idx = np.argsort(lfdr, axis=1)
    bfdr = (np.repeat(1/(np.arange(
        1, lfdr.shape[1] + 1, 1))[:, np.newaxis].transpose(), lfdr.shape[0],
        axis=0))*np.cumsum(np.take_along_axis(lfdr, srt_idx, axis=1), axis=1)
    rej_thr = np.zeros(lfdr.shape[0])
    r = np.zeros(lfdr.shape)
    for mc in np.arange(0, lfdr.shape[0], 1):
        try:
            rej_thr[mc] = np.where(bfdr[mc, :] <= alp)[0][-1]
        except IndexError:
            rej_thr[mc] = -1
        r[mc, srt_idx[mc, np.arange(0, int(rej_thr[mc]+1), 1)].astype(int)] = 1
    return r


def dis_bh(y, gam, bit_bdg):
    """
    Apply the distributed BH procedure from [Ermis2010].

    Parameters
    ----------
    y : numpy array nMC x number sensors
        The test statistics. Must be montonoically decreasing AND uniformly
        distributed under the null hypothesis. We use p-values.
    gam : float
        The nominal FDR level.
    bit_bdg : int
        The bit budget, aka the absolute number of 1-bit transmissions allowed.
        Also equivalent to the number of times the transmitter has to be turned
        on.

    Returns
    -------
    H_s : boolean area, nMC x number sensors
        The descisions.
    it : int vector, nMC x 1
        The number of iterations.

    @author: Martin Goelz
    """
    (nMC, m) = y.shape

    xi = np.ones((nMC, m), dtype=bool)
    H_s = np.zeros((nMC, m), dtype=bool)

    i = np.zeros((nMC, m), dtype=int)
    count = np.zeros((nMC, m), dtype=int)

    phi = np.zeros((nMC, m, m), dtype=bool)
    r = np.zeros((nMC, m), dtype=int)
    it = np.zeros((nMC), dtype=int)

    for mc in np.arange(nMC):
        i[mc, 0] = 1
        t = 1
        go_on = True
        avlbl_bit_bdg = (bit_bdg)
        while go_on:
            l = i[mc, t-1] * gam/m
            H_s[mc, :] = y[mc, :] <= l
            if avlbl_bit_bdg == 0:
                it[mc] = t-1
                break
            phi[mc, :, t-1] = H_s[mc, :]
            announce = xi[mc, :] * phi[mc, :, t-1]

            if np.sum(announce) <= avlbl_bit_bdg:
                xi[mc, announce] = 0
                r[mc, t-1] = np.sum(announce)
            else:
                idx = np.where(announce)[0]
                xi[mc, idx[0:avlbl_bit_bdg]] = 0
                r[mc, t-1] = np.sum(announce[idx[0:avlbl_bit_bdg]])
            avlbl_bit_bdg = avlbl_bit_bdg - r[mc, t-1]
            i[mc, t] = (i[mc, t-1] + r[mc, t-1])
            count[mc, t] = (count[mc, t-1] + r[mc, t-1])
            if (i[mc, t-1] == m or r[mc, t-1] == 0):
                go_on = False
                it[mc] = t
            else:
                t = t + 1
    return H_s, it

# %% Single-sensor control procedure
def p_thr(p, alp):
    """
    Simple hypothesis p-value thresholding (Does not control any
    multiple hypothesis testing quantity).

    Parameters
    ----------
    p : array
        The MC x N matrix of p-values.
            - MC:   # of MC runs
            - N:    # of tested hypotheses
    alp : float
        The FWER nominal level.

    Returns
    -------
    r : array
        The MC x N indicator matrix with an element = 1 if the corresponding
        hypothesis is rejected.

    @Martin Goelz
    """

    r = p <= (alp + np.zeros(p.shape))

    return r
