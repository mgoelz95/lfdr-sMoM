#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides functionalities to deal with spatial fields.

@author: Martin Goelz
"""
# %% imports
import os
import sys

import warnings

from abc import ABC

import numpy as np
from matplotlib import colors

from scipy import constants
from scipy import stats

import pandas as pd

import tuda_colors
import load_and_save as ls

# %% constants
#The color dictionairy for my linearly spaced color-map emphasizing small
#p-values. Setup such that yellow is exactly at 0.15
color_dic = {'red':   [(0.0,  0.0, 1.0),
                       (0.15, 1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             'green': [(0.0,  0.0, 0.0),
                       (0.15, 1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'blue':  [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}

#Alternative colormap fading from red (small p-values) to white instead of
#green (might be needed for colorblind people).
color_dic_ry = {'red':   [(0.0,  0.0, 1.0),
                          (0.15,  1.0, 1.0),
                          (1.0,  1.0, 0.0)],

                'green':  [(0.0,  0.0, 0.0),
                           (0.15,  1.0, 1.0),
                           (1.0,  1.0, 0.0)],

                 'blue':  [(0.0,  0.0, 0.0),
                           (1.0,  1.0, 0.0)]}

cm_fal_dis = colors.LinearSegmentedColormap.from_list(
    'fal_dis',['white', tuda_colors.TUDa_9b],2)
cm_mis_dis = colors.LinearSegmentedColormap.from_list(
    'cor_dis',['white','#555555'],2)
cm_cor_dis = colors.LinearSegmentedColormap.from_list(
    'fal_dis',['white', tuda_colors.TUDa_4b],2)
cm_nul_alt = colors.LinearSegmentedColormap.from_list(
    'nulls_and_alternatives',['white', tuda_colors.TUDa_2b],2)

# %% classes
class Scenario(ABC):
    """
    Abstract base class for multiple hypothesis testing scenarios.

    @author: Martin Goelz
    """
    def __init__(self, n, n_MC):
        self.n = n
        self.n_MC = n_MC
        self.pi0 = np.zeros(n_MC)
        try:
            self.n_0 = (np.round(self.n*self.pi0)).astype(int)
            self.n_1 = self.n-self.n_0
            self.r_tru = np.concatenate((np.zeros((self.n_MC, self.n_0[0])),
                                         np.ones((self.n_MC, self.n_1[0]))),
                                        axis=1)
        except AttributeError:
            print("Invalid initalization parameters!")

    @property
    def n(self):
        """
        The number hypotheses. Has to be a positive integer.
        """

        return self._n

    @property
    def pi0_des(self):
        """
        The fraction of true null hypotheses. Has to be between 0 and 1.
        """
        return self._pi0_des

    @property
    def n_MC(self):
        """
        The number of Monte Carlo runs. Has to be a positive integer.
        """
        return self._n_MC

    @n.setter
    def n(self, val):
        if not np.issubdtype(type(val), np.integer):
            print('Was given non-int number of hypotheses!')
        if val > 0:
            self._n = int(val)
        else:
            print('Hand over a positive number of hypotheses to be tested!')

    @pi0_des.setter
    def pi0_des(self, val):
        if np.ndim(val) == 0:
            if 0 <= val <= 1:
                self._pi0_des = val
            else:
                print('pi_0 not between 0 and 1!')
        else:
            if np.all(np.logical_and(val >= 0, val <= 1)):
                self._pi0_des = val
            else:
                print('At least one pi_0 not between 0 and 1!')

    @n_MC.setter
    def n_MC(self, val):
        if not isinstance(val, int):
            print('Was given non-int number of MC runs!')
        if val > 0:
            self._n_MC = int(val)
        else:
            print('Hand over a positive number of MC runs!')

    @n.deleter
    def n(self):
        del self._n

    @pi0_des.deleter
    def pi0_des(self):
        del self._pi0_des

    @n_MC.deleter
    def n_MC(self):
        del self._n_MC

class SpatialField(Scenario):
    """
    The objects of this class represent a spatial field with a grid, spatial
    units, sensors and occuring alternative events. The field is initialized
    in its null state.

    @author: Martin Goelz
    """
    def __init__(self, scen, dim, n_MC, n_tr_sam, n_obs_sam):
        super().__init__(np.prod(dim), n_MC)

        self.scen = scen
        self.dim = dim
        self.n_tr_sam = n_tr_sam
        self.n_obs_sam = n_obs_sam

        self.n_src = 0  # Initialize number of active transmitters
        self.tot_n_sen = 0  # Initialize total number of sensors
        self.n_sam_per_sen = 0  # Initialize number of observation samples per
        # sensor
        self.n_obs_per_sen = 0  # Initialize observation samples per sensor

        # Initializations of grid points and p-vals
        self.grd_pt = []
        self.p = 0
        self.z = 0

        #Generation of grid points
        self.gen_grd_pt()
        self.gp_lng = 0  # The length of grid points' edges in this field

    def crd_to_sgl_id_2D(self, crd):
        """
        Compute the singular index of the given coordinate w.r.t the spatial
        field.

        Parameters
        ----------
        crd : tuple
            The 2D coordinates. Either a scalar tuple or a tuple of arrays.

        Returns
        -------
        int
            The singular index of this coordinate.

        @author: Martin Goelz
        """

        return self.dim[0]*crd[0] + crd[1]

    def con_gp_alt_evt(self):
        """Contaminate field with alternative event.
        """
        print("This method does not have any functionality here!")

    def gen_grd_pt(self):
        """
        Generate pixels for this spatial field. Supports max. 3D coordinates
        due to required nested for-loops, the absolute number of layers, aka
        spatial dimensions, has to be known upfront.

        Returns
        -------
        None.

        @author: Martin Goelz
        """
        sgl_id = 0
        if np.array(self.dim).size == 1:
            crd_rge = np.arange(0, self.dim, 1)
            for crd in crd_rge:
                self.grd_pt.append(
                    GridPoint(sgl_id, crd, (np.zeros((self.n_MC, 1))),
                    self.n_MC, self.n_tr_sam, self.n_obs_sam))
                print(f'\rGenerated grid point: {sgl_id+1}/{self.n}', end="")
                sgl_id += 1
            print('')
        elif np.array(self.dim).size == 2:
            x_rge = np.arange(0, self.dim[0], 1)
            y_rge = np.arange(0, self.dim[1], 1)
            sgl_id = 0
            for y in y_rge:
                for x in x_rge:
                    self.grd_pt.append(GridPoint(sgl_id, ((y, x)),
                                                   np.zeros((self.n_MC, 1)),
                                                   self.n_MC, self.n_tr_sam,
                                                   self.n_obs_sam))
                    print(f'\rGenerated grid point: {sgl_id+1}/{self.n}',
                    end="")
                    sgl_id += 1
                #gc.collect() #Use only if really necessary!
            print('')
        elif np.array(self.dim).size == 3:
            x_rge = np.arange(0, self.dim[0], 1)
            y_rge = np.arange(0, self.dim[1], 1)
            z_rge = np.arange(0, self.dim[2], 1)
            for z in z_rge:
                for y in y_rge:
                    for x in x_rge:
                        self.grd_pt.append(
                            GridPoint(sgl_id, ((z, y, x)),
                                        np.zeros((self.n_MC, 1)), self.n_MC,
                                         self.n_tr_sam, self.n_obs_sam))
                        print(f'\rGenerated grid point: {sgl_id+1}/{self.n}',
                              end="")
                        sgl_id += 1
            print('\n')

    def res_fd(self):
        """Reset the field.
        """
        print("This has no functionality here!")

    def sgl_id_to_crd_2D(self, sgl_id):
        """
        Compute the 2D coordinate of a given singular index w.r.t the spatial
        field.

        Parameters
        ----------
        sgl_id : array
            The singular indeces to convert. Either scalar or 1D-array

        Returns
        -------
        tuple
            The corresponding 2D coordinates. Either a scalar or array tuple.

        @author: Martin Goelz
        """

        return (np.array((sgl_id/self.dim[0])).astype(int),
                np.remainder(sgl_id, self.dim[0]))

    def upd_nul_pro(self, pi0, n_0=-1):
        """
        Update the null proportion and set the number of null and alternative
        grid points accordingly. Only to be called in combination with a
        function that updates the p-values of the field accordingly! If a
        number of nulls is given instead of a null fraction, update the
        fraction accordingly.

        Parameters
        ----------
        pi0 : float
            The true proportion of grid points at which H_0 is in place.

        n_0 : int
            The number of grid points at which H_0 is in place. If -1, we
            work with the given proportion. The default is -1.

        Returns
        -------
        None.

        @author: Martin Goelz§
        """

        if np.ndim(n_0) == 0 and n_0 == -1:
            self.pi0 = pi0 + np.zeros(self.n_MC)
            self.n_0 = np.array([np.round(self.n*self.pi0)]).astype(int)
            self.n_1 = self.n-self.n_0
        else:
            self.n_0 = n_0
            self.n_1 = self.n - self.n_0
            self.pi0 = self.n_0/self.n

    def upd_tst(self):
        """
        Copy the p-values and z-scores at the individual grid points into
        the field-level matrix of p-values and z-scores. Always to be called
        after changes to the grid point p-vals and z-scores have been made!

        Returns
        -------
        None.

        @author: Martin Goelz
        """

        self.p = np.array([gp.p for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()
        self.z = np.array([gp.z for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()

    @property
    def sen_idx(self):
        """
        The matrix containing the pixel indexes of sensors.
        """
        return self._sen_idx

    @sen_idx.setter
    def sen_idx(self, val):
        self._sen_idx = val
        self.tot_n_sen = val.shape[1]

    @sen_idx.deleter
    def sen_idx(self):
        del self._sen_idx

class CustomSpatialField(SpatialField):
    def __init__(self, fd_scen, dat_path):
        custom = pd.read_pickle(os.path.join(dat_path, fd_scen + '.pkl'))
        super().__init__(
            fd_scen, custom['fd_dim'][0], custom['p'][0].shape[0], 0, 0)
        self.p = np.zeros((self.n_MC, self.n)) + np.nan
        self.z = np.zeros((self.n_MC, self.n)) + np.nan
        self.r_tru = np.zeros((self.n_MC, self.n)) + np.nan
        self.sen_cds = custom['sen_cds'][0]
        self.sen_idx = np.zeros(custom['p'][0].shape, dtype=int)
        for mc in np.arange(self.n_MC):
            for idx in np.arange(self.sen_idx.shape[1]):
                self.sen_idx[mc, idx] = self.crd_to_sgl_id_2D(
                    self.sen_cds[mc, idx, [1, 0]])
        for mc in np.arange(self.n_MC):
            self.p[mc, self.sen_idx[mc, :]] = custom['p'][0][mc, :]
            self.z[mc, self.sen_idx[mc, :]] = stats.norm.ppf(
                self.p[mc, self.sen_idx[mc, :]])
            try:
                self.r_tru[mc, self.sen_idx[mc, :]] = custom['r_tru'][0][mc, :]
            except KeyError:
                self.r_tru[mc, self.sen_idx[mc, :]] = np.nan
        self.res_fd()

    def res_fd(self):
        for gp in self.grd_pt:
            gp.X_0 = None
            gp.Tau_0 = None
            gp.x_tr = None
            gp.X_1 = None
            gp.tau = None
            gp.p = np.zeros(self.n_MC) + np.nan
            gp.z = np.zeros(self.n_MC) + np.nan
            gp.h = np.zeros(self.n_MC) + np.nan
            for mc in np.arange(self.n_MC):
                idx = np.where(self.sen_idx[mc, :] == gp.sgl_id)[0]
                if not idx.size == 0:
                    gp.p[mc] = self.p[mc, gp.sgl_id]
                    gp.z[mc] = self.z[mc, gp.sgl_id]
                    gp.h[mc] = self.r_tru[mc, gp.sgl_id]

    def upd_nul_pro(self, pi0, n_0=-1):
        print("This method doesn't do anything for CustomSpatialFields.")

    def upd_tst(self):
        print("This method doesn't do anything for CustomSpatialFields.")

class CustomSpatialFieldEstimated(object):

    def __init__(self, fd):
        if isinstance(fd, CustomSpatialField):
            self.n_MC = fd.n_MC
            self.dim = fd.dim
            self.gp_lng = fd.gp_lng
            self.n_gp = fd.n
            self.n = fd.sen_idx.shape[1]
            self.scen = fd.scen + '_est'
            self.p = np.nan + np.zeros((self.n_MC, self.n))
            self.z = np.nan + np.zeros((self.n_MC, self.n))
            self.p_gp = np.nan + np.zeros((self.n_MC, self.n_gp))
            self.z_gp = np.nan + np.zeros((self.n_MC, self.n_gp))

            self.sen_idx = fd.sen_idx
            self.r_tru = np.take_along_axis(
                fd.r_tru, fd.sen_idx, axis=1)

            for idx in np.arange(0, self.n_MC, 1):
                self.p[idx, :] = fd.p[
                    idx, self.sen_idx[idx, :]]
                self.p_gp[idx, self.sen_idx[idx, :]] = fd.p[
                    idx, self.sen_idx[idx, :]]
                self.z_gp[idx, self.sen_idx[idx, :]] = fd.z[
                    idx, self.sen_idx[idx, :]]
                self.z[idx, :] = fd.z[
                    idx, self.sen_idx[idx, :]]
            self.grd_pt = []
            self.gen_grd_pt()
            self.upd_grd_pt()

            self.sen_cds = fd.sen_cds
        else:
            print("Requires an object of CustomSpatialField!")

    def crd_to_sgl_id_2D(self, crd):
        """
        Compute the singular index of the given crd w.r.t the spatial field.

        Parameters
        ----------
        crd : tuple
            The 2D coordinates. Either a scalar tuple or a tuple of arrays.

        Returns
        -------
        int
            The singular index of this coordinate.

        @author: Martin Goelz
        """
        return self.dim[0]*crd[0] + crd[1]

    def gen_grd_pt(self):
        """
        Generate grid points for this estimated spatial field.

        Returns
        -------
        None.

        @author: Martin Goelz
        """
        sgl_id = 0

        x_rge = np.arange(0, self.dim[0], 1)
        y_rge = np.arange(0, self.dim[1], 1)
        sgl_id = 0
        for y in y_rge:
            for x in x_rge:
                self.grd_pt.append(GridPointEstimated(
                    sgl_id, ((y, x)), self.n_MC, np.nan + np.zeros(self.n_MC),
                    np.nan + np.zeros(self.n_MC), np.nan + np.zeros(self.n_MC),
                    np.nan + np.zeros(self.n_MC)))
                print(f'\rGenerated grid point: {sgl_id+1}/{self.n_gp}',
                end="")
                sgl_id += 1
        print('\rGeneration of estimated spatial field completed!')

    def quantize_p_and_z(self, q_borders, q_centers):
        """Quantize p-values and z-scores

        Parameters
        ----------
        q_borders : numpy array
            The edges of the quantization intervals.
        q_centers : numpy array
            The centers of the quantization intervals.
        """
        quan_idx = np.zeros(self.p.shape, dtype=int)
        quan_vls = np.zeros(self.p.shape)
        for mc in np.arange(self.p.shape[0]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in divide")
                quan_idx_mc = np.nanargmin(
                    (self.p[mc, :] / np.tile(q_borders[:, np.newaxis],
                                             [self.p[mc, :].size])) - 1 > 0,
                    0) - 1
            quan_vls_mc = q_centers[quan_idx_mc]
            quan_idx[mc, :] = quan_idx_mc
            quan_vls[mc, :] = quan_vls_mc
            quan_vls[mc, np.isnan(self.p[mc, :])] = np.nan
        self.p = quan_vls
        self.quan_idx = quan_idx
        self.z = stats.norm.ppf(self.p)

    def sgl_id_to_crd_2D(self, sgl_id):
        """
        Compute the 2D crd of a given singular index w.r.t the spatial field.

        Parameters
        ----------
        sgl_id : array
            The singular indeces to convert. Either scalar or 1D-array

        Returns
        -------
        tuple
            The corresponding 2D coordinates. Either a scalar or array tuple.

        @author: Martin Goelz
        """
        return (np.array((sgl_id/self.dim[0])).astype(int),
                np.remainder(sgl_id, self.dim[0]))

    def upd_grd_pt(self):
        """
        Update the grid points of the field.

        Copying the variables that were updated on the field level to the
        grid points.

        Returns
        -------
        None.

        @author: Martin Goelz
        """
        for gp in self.grd_pt:
            for mc in np.arange(self.n_MC):
                idx = np.where(self.sen_idx[mc, :] == gp.sgl_id)[0]
                if not idx.size == 0:
                    gp.p[mc] = self.p[mc, idx]
                    gp.z[mc] = self.z[mc, idx]
                    gp.sen_idc[mc] = ~np.isnan(self.p[mc, idx])
                else:
                    gp.p[mc] = np.nan
                    gp.z[mc] = np.nan
                    gp.sen_idc[mc] = np.nan


class FakeField(Scenario):
    def __init__(self, fd_scen, dat_path):
        custom = pd.read_pickle(os.path.join(dat_path, fd_scen + '.pkl'))
        super().__init__(custom['p'][0].shape[1], custom['p'][0].shape[0])

        self.scen = fd_scen

        self.p = custom['p'][0]
        self.z = stats.norm.ppf(self.p)
        try:
            self.r_tru = custom['r_tru'][0]
        except KeyError:
            self.r_tru == np.zeros(self.p.shape) + np.nan
    
class RadioSpatialField(SpatialField):
    """
    The objects of this class represent an artificial spatial field with a
    grid, spatial. What distincts a RadioSpatialField from other objects
    of class SpatialField is the fact that the field values at different
    grid points are generated using an accurate free-space propagation model.
    Then, an energy detector is deployed on a single grid point level and
    p-values are computed accordingly.

    @author: Martin Goelz
    """
    def __init__(self, scen, dim, n_MC, n_tr_sam, n_obs_sam):
        if len(dim) != 2:
            print("A radio field has to be 2D!")
        else:
            super().__init__(scen, dim, n_MC, n_tr_sam, n_obs_sam)
        self.res_fd()

        """Remarks on the specification of the alternative events:
            A: num of event params x num of alt events x num MC runs
              "The specification matrix":
                This matrix summerizes the alternative event parameters for
                all clusters and all MC runs. The first parameter (aka
                A[0, :, :]) is ALWAYS the number of sensors affected by this
                event. The last parameter (aka A[-1, :, :]) is ALWAYS the
                beta distribution parameter.
            Q: num MC runs x num of grid points x num of alt events
              "The affection matrix":
                The association of each grid point with an event. If Q[m,n,k]
                = 1, then the grid point n is at MC run m associated with the
                alternative event k. If Q[m,n,k] = 0 for all k = 1, ..., n_src,
                H0 is true. In other words, H_1 is true wherever sum(Q[m,n,:])
                >0.
            X: num MC runs x num of grid points
              "The radio field level matrix"
               Captures the observed field level at each individual spatial
               unit before the addition of noise, which is added only on the
               grid point level."""

        self.A = np.zeros((2, self.n_src, self.n_MC))
        self.Q = np.zeros((self.n_MC, self.n, self.n_src), dtype=int)
        self.X = np.zeros((self.n_MC, self.n))

    def com_gp_lng(self, D, s_t_dB, s_r_min_dB, K_dB, sig_s):
        """
        Compute side-length of a grid point such that a single source would
        cover self.n_1/self.n_src number of pixels if that source was placed at
        the center of the field.

        The percentage of alternative sensors can also be obtained almost
        perfectly with this method for an arbitrary location of the source.
        Works only for a single source, however. To obtain this, provide a D
        that summarizes the distance between the location of the source and
        the grid points (instead of the distances to the center of the map).
        Also make sure that n_src is set to 1. The resultung fraction of
        grid points with H1 in place is then pretty accurately equal to the
        desired one. Set D = d_gp_cen[:,:,k,mc] from self.pla_tra() to obtain
        this.

        Parameters
        ----------
        D : array
            The M x N grid of distances to the transmitter (M being the number
            of grid points in y-direction and N in x-direction).
        s_t_dB : float
            The transmit power of the source, in dB.
        s_r_min_dB : float
            The minimum received signal power for a receiver to be declared
            as true H_1.
        K_dB : float
            The constant free-space path-loss coefficient, in dB.
        sig_s : TYPE
            The shadow-fading variance, in dB.

        @author: Martin Goelz
        """

        num_can = 40
        D[np.where(D==0)] = 1/np.sqrt(2)
        gp_lng_can = np.logspace(-1, 1.5, base=10, num=num_can)

        rv = np.zeros((self.n, num_can))
        for idx in np.arange(0, num_can):
            rv[:, idx] = stats.norm(loc = -10*np.log10(
                gp_lng_can[idx]) + (
                    s_t_dB + K_dB - 10*np.log10(D)).reshape(self.n),
                    scale = sig_s).cdf(s_r_min_dB)
        p_can = (1-np.mean(rv, axis=0))
        gp_lng = gp_lng_can[np.argmin(
            np.abs(p_can-(1-self.pi0[0])/self.n_src))]

        self.gp_lng = gp_lng

    def con_gp_alt_evt(self):
        """
        Contaminate the grid points according to the specified affection
        matrix Q and the specification of the alternative events in A.
        re_obs this modified field for at each grid point. Update
        the spatial field p-value matrix accordingly. Basically gives the
        field values stored in self.X down to every grid point.

        @author Martin Goelz
        """

        for gp in self.grd_pt:
            gp.rad_fd_up_val(self.X[:, gp.sgl_id])
            print(f'\rUpdated grid point: {gp.sgl_id+1}/{self.n}', end="")
        print('')
        self.upd_tst()
        self.r_tru = self.X > 0

    def pla_tra(self, fd_scen, dat_path):
        """
        Place a given number n_src of transmitters within the scope of the
        field.
        The transmitters are all affecting elipses of grid points. Both,
        radius and centers of the alternative event affection regions can be
        either homogeneous (equal spacing across field and/or equal size) or
        random. If random, the radii of the events are poisson random variates
        with the rate equal the radius corresponding to equally sized
        alternatives. The centers are uniformly distributed
        across the field. The true fraction of alternatives
        might become much smaller as overlap comes into play, which might be
        critical especially for large pi_1.
        Produce a new affection matrix Q and specification
        matrix A for this spatial field.
        If shadow-fading is simulated, the transmit power is constant for all
        sources, so the variations in the covered areas are a result of the
        shadowing only.

        The specification matrix is of dimension 8 x self.n_src x self.n_MC.
        The entries of the first dimension correspond to:
            0: Number of grid points affected by this event
            1: Index for the central grid point of this event
            2: Radius of this event
            3: Precision matrix left top entryfor this event
            4: Precision matrix right top entry for this event
            5: Precision matrix left bottom entry for this event
            6: Precision matrix right bottom entry for this event
            7: The transmit power for this event (in linear scale).

        Parameters
        ----------
        fd_scen : str
            The scenario name.
        dat_path : str
            The path to where the data is stored.
        @author Martin Goelz
        """
        [fd_dim, n_MC, n_sam, n_src, ran_cen, ran_rad, ran_pre, add_tra,
            pi0_des, sha_fa, prop_env] = ls.ld_sc(dat_path, fd_scen)
        if not ran_pre:
            fix_pre = np.array([[1, 0], [0, 1]])  # Produces circles
        if prop_env == "urb":
            sig_s = 4
        elif prop_env == "suburb":
            sig_s = 8
        if (not fd_dim == self.dim or not n_MC == self.n_MC or
            not n_sam == self.n_obs_sam):
            print("The field does not fit to the scenario! Check whats wrong")
            return

        #Initialization with handed over parameters
        self.upd_nul_pro(pi0_des)
        self.n_src = n_src
        self.A = np.zeros((8, self.n_src, self.n_MC))

        #x and y coordinate range
        x_rge = np.arange(0, self.dim[0], 1)
        y_rge = np.arange(0, self.dim[1], 1)

        #The meshed coordinate grid -> For computation of distances to
        #centers later on needed.
        (X,Y) = np.meshgrid(x_rge, y_rge)

        #Reading in the coordinate from the grid point list
        x_crd = np.array([gp.crd[1] for gp in self.grd_pt])
        y_crd = np.array([gp.crd[0] for gp in self.grd_pt])


        #Signal parameters
        # # # # # # # # # # # # # # # #
        #Minimum signal value to be identified as H_1
        s_r_min = 0.2
        s_r_min_dB = 10*np.log10(s_r_min)
        #The frequency the field is evaluated at
        rad_frq = 2.535*10**9 #Radio frequency
        #Propagation factor in free-space
        K = constants.c/(4*np.pi*rad_frq)
        K_dB = 10*np.log10(K) #in dB
        #Reference transmit power
        s_t_ref_dB = s_r_min_dB + 37 #in dB -> in dependence of the min. rec
        #power (stems from the idea that a receive power ~80dB below transmit
        #can still be detected. So only the difference between s_r_min_dB and
        #s_t_ref_dB matter for the system to be realistic. The actual values
        #are to be determined in relation to the noise at the receivers (SNR)!)
        s_t_ref = 10**(s_t_ref_dB/10)

        #Random radius and computation of size of alternative events based
        #on these radii.
        if ran_rad and not sha_fa:
            rad_ev = (stats.poisson.rvs(
                np.round(np.sqrt((self.n_1[0,0]/self.n_src)/np.pi)), size=(
                    (self.n_MC, self.n_src)))).astype(int)
            n_ev = np.pi + rad_ev**2
            self.gp_lng = s_t_ref/(
                s_r_min*np.sqrt(self.n_1[0,0]/self.n_src/np.pi))*K
        #Fix radius and/or shadow fading active
        else:
            n_ev = self.n_1[0,0]/self.n_src + np.zeros((self.n_MC, self.n_src))
            rad_ev = np.round((np.sqrt(n_ev/(np.pi)))).astype(int)
            if sha_fa:
                self.com_gp_lng(np.sqrt((
                    X-np.take(x_rge, x_rge.size // 2))**2+(Y - np.take(
                        y_rge, y_rge.size // 2))**2), s_t_ref_dB,
                        s_r_min_dB, K_dB, sig_s)
            else:
                self.gp_lng = s_t_ref/(s_r_min*np.sqrt(
                    self.n_1[0,0]/self.n_src/np.pi))*K

        #Determination of transmit power for all transmitters in the field
        s_t = (self.gp_lng*s_r_min * rad_ev/K).transpose()
        s_t_dB = 10*np.log10(s_t) #in dB

        #Storing the powers in the parameter matrix
        self.A[-1, :, :] = s_t_dB#transmit powers
        # # # # # # # # # # # # # # # #

        #Homogeneously spaced centers
        if not ran_cen:
            cen_ev = (np.int(np.floor(self.n/self.n_src/2)+
                             np.int(np.floor(self.dim[0]/2))) +
                          (self.n_1[0,0]/self.n_src + np.zeros(
                              (self.n_MC, self.n_src)).astype(int) +
                              (self.n_0.transpose()/self.n_src))*np.arange(
                              0, self.n_src, 1)).astype(int)
        #Random centers
        else:
            cen_ev = np.floor(stats.uniform.rvs(
                0, self.n, size=(self.n_MC, self.n_src))).astype(int)

        x_cen = x_crd[cen_ev]
        y_cen = y_crd[cen_ev]

        #Initialization of received signal values
        self.X = np.zeros((self.n_MC, self.n))

        #Initialization of precision matrix
        pre_mat = np.zeros((2, 2, self.n_src, self.n_MC))

        #Initializations of distances between grid points and event
        #centers
        dis_gp_cen = np.zeros(
            (self.dim[1], self.dim[0], self.n_src, self.n_MC))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #Initialization of alternative indeces
            alt_ind = np.repeat((np.array([np.empty(gp) for gp in np.arange(
                0, self.n_MC, 1)]))[:, np.newaxis], self.n_src, axis=1)

        #If we want ellipses: Random precision matrices
        if ran_pre:
            ran_mat = np.zeros((2, 2))
            cov = np.zeros((2, 2))
            uni_rnd = stats.uniform

            for mc in np.arange(0, self.n_MC, 1):
                for k in np.arange(0, self.n_src, 1):
                    ran_mat = uni_rnd.rvs(-1, 2, size=(2,2))
                    ran_mat = uni_rnd.rvs(0, .5, size=(
                        2,2)) + .8*np.eye(2)
                    sig_cov = uni_rnd.rvs()>.5
                    ran_mat[[0,1],  [1, 0]] = ran_mat[
                        [0,1],  [1, 0]]*(sig_cov) - ran_mat[
                        [0,1],  [1, 0]]*(not sig_cov)
                    cov = np.matmul(ran_mat, ran_mat.transpose())
                    cov = 1/np.sqrt(np.abs(np.linalg.det(cov))) *cov
                    pre_mat[:, :, k, mc] = 1/(cov[0,0]*cov[1,1] -
                                              cov[1,0]*cov[0,1])*np.array(
                [[cov[1,1], -cov[0,1]],[-cov[1,0], cov[0,0]]])
                    dis_gp_cen[:, :, k, mc]= np.sqrt(((
                        X-x_cen[mc, k])*pre_mat[0,0, k, mc]+(
                            Y-y_cen[mc, k])*pre_mat[0,1, k, mc])*(
                                X-x_cen[mc, k])+((
                                    X-x_cen[mc, k])*pre_mat[1,0, k, mc]+(
                                        Y-y_cen[mc, k])*pre_mat[
                                            1,1, k, mc])*(Y-y_cen[mc, k]))
                    #Replacing the zero-distance at the transmitter location
                    #(to avoid division by 0)
                    dis_gp_cen[:,:,k,mc][
                        np.where(dis_gp_cen[:,:,k,mc]==0)] = 1/np.sqrt(2)
                    sig = (self.A[-1, k, mc] -10*np.log10(self.gp_lng)
                               + K_dB - 10*np.log10(
                            dis_gp_cen[: , :, k, mc].reshape(self.n)))
                    if not sha_fa:#no shadow fading
                        if add_tra:#the received signals add up
                            self.X[mc, :] += 10**((sig)/10)
                            alt_ind[mc, k] = np.where(
                                self.X[mc, :]>= s_r_min)[0]
                        else:#the received signals do not add up
                            can_2D = (np.where(np.all(
                                [dis_gp_cen[:,:,k,mc] <= rad_ev[mc, k],
                                  X>= 0, Y >= 0],
                                axis=0)))
                            alt_ind[mc,k] =  self.crd_to_sgl_id_2D(
                                (Y[can_2D], X[can_2D]))
                            self.X[mc, alt_ind[mc,k]] += 10**((self.A[
                                  -1, k, mc] + K_dB - 10*np.log10(dis_gp_cen[
                                can_2D[0], can_2D[1], k, mc])- 0)/10)
                    else:#shadow fading
                        sf =(self.sha_fa(
                            x_rge, y_rge, x_crd[cen_ev[mc, k]],
                            y_crd[cen_ev[mc, k]], sig_s).reshape(self.n))
                        if add_tra:#the received signals add up
                            self.X[mc, :] += 10**((sig + sf)/10)
                            alt_ind[mc, k] = np.where(
                                self.X[mc, :] >= s_r_min)[0]
                        else:#the received signals do not add up
                            rec_sig = 10**((sig + sf)/10)
                            alt_ind[mc, k] = np.where(rec_sig >= s_r_min)[0]
                            self.X[mc, alt_ind[mc,k]] = np.max(
                                (rec_sig[alt_ind[mc, k]],
                                self.X[mc, alt_ind[mc,k]]), axis=0)
                #Nulling the signal below the minimum received power to obtain
                #the true detection pattern. Needed only for add_tra = true.
                self.X[mc, np.where(self.X[mc, :]<s_r_min)[0]] = 0
                print(f'\rSimulated field for MC run: {mc+1}/{self.n_MC}',
                end="")
            print('')
        #If we want circles: Fixed precision matrix
        else:
            pre_mat = np.repeat(
                np.repeat(fix_pre[:, :, np.newaxis],
                      self.n_src, axis = 2)[:, :, :,
                            np.newaxis],
                                self.n_MC, axis = 3)
            # def sgl_mc_run(mc):
            for mc in np.arange(self.n_MC):
                for k in np.arange(0, self.n_src, 1):
                    dis_gp_cen[:, :, k, mc]= np.sqrt(((
                        X-x_cen[mc, k])*pre_mat[0,0, k, mc]+(
                            Y-y_cen[mc, k])*pre_mat[0,1, k, mc])*(
                                X-x_cen[mc, k])+((
                                    X-x_cen[mc, k])*pre_mat[1,0, k, mc]+(
                                        Y-y_cen[mc, k])*pre_mat[
                                            1,1, k, mc])*(
                                            Y-y_cen[mc, k]))
                    #Replacing the zero-distance at the transmitter location
                    #(to avoid division by 0)
                    dis_gp_cen[:,:,k,mc][
                        np.where(dis_gp_cen[:,:,k,mc]==0)] = 1/np.sqrt(2)
                    sig = (self.A[-1, k, mc] -10*np.log10(self.gp_lng)
                               + K_dB - 10*np.log10(
                            dis_gp_cen[: , :, k, mc].reshape(self.n)))
                    if not sha_fa:#no shadow fading
                        if add_tra:#the received signals add up
                            self.X[mc, :] += 10**((sig)/10)
                            alt_ind[mc, k] = np.where(
                                self.X[mc, :]>= s_r_min)[0]
                        else:#the received signals do not add up
                            can_2D = (np.where(np.all(
                                [dis_gp_cen[:,:,k,mc] <= rad_ev[mc, k],
                                  X>= 0, Y >= 0],
                                axis=0)))
                            alt_ind[mc,k] =  self.crd_to_sgl_id_2D(
                                (Y[can_2D], X[can_2D]))
                            self.X[mc, alt_ind[mc,k]] += 10**((self.A[
                                  -1, k, mc] + K_dB - 10*np.log10(dis_gp_cen[
                                can_2D[0], can_2D[1], k, mc])- 0)/10)
                    else:#shadow fading
                        sf =(self.sha_fa(
                            x_rge, y_rge, x_crd[cen_ev[mc, k]],
                            y_crd[cen_ev[mc, k]], sig_s).reshape(self.n))
                        if add_tra:#the received signals add up
                            self.X[mc, :] += 10**((sig + sf)/10)
                            alt_ind[mc, k] = np.where(
                                self.X[mc, :] >= s_r_min)[0]
                        else:#the received signals do not add up
                            rec_sig = 10**((sig + sf)/10)
                            alt_ind[mc, k] = np.where(rec_sig >= s_r_min)[0]
                            self.X[mc, alt_ind[mc,k]] = np.max(
                                (rec_sig[alt_ind[mc, k]],
                                self.X[mc, alt_ind[mc,k]]), axis=0)
                #Nulling the signal below the minimum received power to obtain
                #the true detection pattern. Needed only for add_tra = true.
                self.X[mc, np.where(self.X[mc, :]<s_r_min)[0]] = 0
                print(f'\rSimulated field for MC run: {mc+1}/{self.n_MC}',
                end="")
            # par_pl = mp.Pool(np.min((50, os.cpu_count() - 1)))
            # par_pl.map(sgl_mc_run, np.arange(self.n_MC))
            # par_pl.close()
            print('')
        self.Q = np.zeros((self.n_MC, self.n, self.n_src))
        for k in np.arange(1, self.n_src+1, 1):
            for mc in np.arange(0,self.n_MC, 1):
                self.Q[mc, alt_ind[mc,k-1], k-1] = 1

        #The actual size of alternative events
        n_ev = np.sum(self.Q==1, axis=1)
        #The actual gobal null proportion
        self.upd_nul_pro(0, n_0=self.n-np.sum(
            np.sum(self.Q==1, axis=2)>0, axis=1))

        self.A[0, :, :] = n_ev.transpose()
        self.A[1, :, :] = cen_ev.transpose()
        self.A[2, :, :] = rad_ev.transpose()
        self.A[3, :, :] = pre_mat[0, 0, :, :]#.transpose()
        self.A[4, :, :] = pre_mat[0, 1, :, :]#.transpose()
        self.A[5, :, :] = pre_mat[1, 0, :, :]#.transpose()
        self.A[6, :, :] = pre_mat[1, 1, :, :]#.transpose()

        self.con_gp_alt_evt()

    def pla_sen(self, dat_path, sen_cfg):
        """Place sensors in the field.

        Parameters
        ----------
        dat_path : str
            The path to where the data is stored.
        sen_cfg : str
            The sensor configuration name.
        """
        n_sen, n_obs_per_sen, var_sen_loc, sen_hom = ls.ld_cfg(
            dat_path, sen_cfg)
        self.n_sen = n_sen
        self.n_obs_per_sen = n_obs_per_sen
        if np.isscalar(n_sen):
            if var_sen_loc:
                sen_idx = np.zeros((self.n_MC, n_sen)).astype(int)
                if sen_hom:
                    for mc in np.arange(self.n_MC):
                        sen_idx[mc, :] = np.random.choice(
                            np.prod(self.dim), n_sen, replace=False)
                else:
                    num_sen_sec_1 = int((n_sen/3)*2)
                    num_sen_sec_2 = int((n_sen/3)*0.8)
                    num_sen_sec_3 = n_sen - num_sen_sec_1 - num_sen_sec_2
                    for mc in np.arange(self.n_MC):
                        sen_idx[mc, :] = np.concatenate(
                            [np.random.choice(
                                np.arange(0, int(np.prod(self.dim)/3)),
                                num_sen_sec_1, replace=False),
                             np.random.choice(
                                 np.arange(
                                     int(np.prod(self.dim)/3), int(2*np.prod(
                                         self.dim)/3)), num_sen_sec_2,
                                 replace=False),
                             np.random.choice(
                                 np.arange(
                                     2*int(np.prod(self.dim)/3), int(np.prod(
                                         self.dim))), num_sen_sec_3,
                                 replace=False)])
            else:
                if sen_hom:
                    sen_idx = (np.random.choice(
                        np.prod(self.dim), n_sen, replace=False)
                        + np.zeros((self.n_MC, n_sen), dtype=int))
                else:
                    num_sen_sec_1 = int((n_sen/3)*2)
                    num_sen_sec_2 = int((n_sen/3)*0.8)
                    num_sen_sec_3 = n_sen - num_sen_sec_1 - num_sen_sec_2
                    sen_idx = np.concatenate(
                        [np.random.choice(
                            np.arange(0, int(np.prod(self.dim)/3)),
                            num_sen_sec_1, replace=False),
                         np.random.choice(
                             np.arange(int(np.prod(self.dim)/3),
                                       int(2*np.prod(self.dim)/3)),
                             num_sen_sec_2, replace=False),
                         np.random.choice(
                             np.arange(2*int(np.prod(self.dim)/3),
                                       int(np.prod(self.dim))),
                             num_sen_sec_3, replace=False)]) + np.zeros((
                                              self.n_MC, n_sen), dtype=int)
            self.sen_idx = sen_idx
        else:
            sen_idx = []
            num_sen_types = n_sen.size
            all_indeces = np.zeros((self.n_MC, np.sum(n_sen))).astype(int)
            if np.sum(n_sen == self.n) > 0:
                all_indeces = np.tile(np.arange(self.n)[np.newaxis],
                    (self.n_MC, 1))
                sen_idx.append(all_indeces)
            else:
                for sen_type in np.arange(num_sen_types):
                    prev_num_sen = np.sum(n_sen[0:np.max([0, sen_type])])
                    if var_sen_loc[sen_type]:
                        sen_idx.append(
                            np.zeros((self.n_MC, n_sen[sen_type])).astype(int))
                        if sen_hom[sen_type]:
                            for mc in np.arange(self.n_MC):
                                sen_idx[sen_type][mc, :] = np.random.choice(
                                    np.setdiff1d(
                                        np.arange(
                                            np.prod(self.dim)),
                                        all_indeces[mc, :]),
                                    n_sen[sen_type], replace=False).astype(int)
                                all_indeces[
                                    mc, prev_num_sen:prev_num_sen+n_sen[
                                        sen_type]] = sen_idx[sen_type][mc, :]
                        else:
                            num_sen_sec_1 = int((n_sen[sen_type]/3)*2)
                            num_sen_sec_2 = int((n_sen[sen_type]/3)*0.8)
                            num_sen_sec_3 = (
                                n_sen[sen_type] - num_sen_sec_1
                                - num_sen_sec_2)
                            for mc in np.arange(self.n_MC):
                                one = np.random.choice(
                                    np.setdiff1d(
                                        np.arange(0, int(np.prod(self.dim)/3)),
                                        all_indeces[mc, :]),
                                    num_sen_sec_1, replace=False)
                                all_indeces[mc,
                                            prev_num_sen:prev_num_sen + int(
                                                np.prod(self.dim)/3)] = one
                                two = np.random.choice(
                                    np.setdiff1d(
                                        np.arange(
                                            int(np.prod(self.dim)/3),
                                            int(2*np.prod(self.dim)/3)),
                                        all_indeces[mc, :]),
                                    num_sen_sec_2, replace=False)
                                all_indeces[
                                    mc, prev_num_sen + np.arange(int(
                                        np.prod(self.dim)/3), int(2*np.prod(
                                         self.dim)/3))] = two
                                three = np.random.choice(
                                    np.setdiff1d(
                                        np.arange(
                                            2*int(np.prod(self.dim)/3),
                                            int(np.prod(
                                                self.dim))),
                                        all_indeces[mc, :]), num_sen_sec_3,
                                    replace=False)
                                all_indeces[mc, (prev_num_sen
                                    + 2*int(np.prod(self.dim)/3)):] = three
                                sen_idx[sen_type][mc, :] = np.concatenate(
                                    [one, two, three])
                    else:
                        if sen_hom[sen_type]:
                            sen_idx.append(
                                np.random.choice(
                                    np.setdiff1d(
                                        np.arange(np.prod(self.dim)),
                                        all_indeces[mc, :]),
                                    n_sen[sen_type],
                                    replace=False) + np.zeros(
                                        (self.n_MC, n_sen[sen_type]),
                                        dtype=int))
                            all_indeces[
                                :, np.sum(
                                    n_sen[0:np.max(0, sen_type-1)]):n_sen[
                                            sen_type]] = np.tile(
                                                sen_idx[sen_type][:],
                                                [self.n_MC, 1])
                        else:
                            num_sen_sec_1 = int((n_sen[sen_type]/3)*2)
                            num_sen_sec_2 = int((n_sen[sen_type]/3)*0.8)
                            num_sen_sec_3 = (
                                n_sen[sen_type] - num_sen_sec_1
                                - num_sen_sec_2)
                            one = np.random.choice(
                                np.setdiff1d(
                                    np.arange(0, int(np.prod(self.dim)/3)),
                                    all_indeces[mc, :]),
                                num_sen_sec_1, replace=False)
                            all_indeces[:,
                                        prev_num_sen:prev_num_sen + int(
                                            np.prod(self.dim)/3)] = np.tile(
                                                one, [self.n_MC, 1])
                            two = np.random.choice(
                                np.setdiff1d(
                                    np.arange(
                                        int(np.prod(self.dim)/3),
                                        int(2*np.prod(self.dim)/3)),
                                    all_indeces[mc, :]),
                                num_sen_sec_2, replace=False)
                            all_indeces[
                                :, prev_num_sen + np.arange(int(
                                    np.prod(self.dim)/3), int(2*np.prod(
                                     self.dim)/3))] = np.tile(two,
                                     [self.n_MC, 1])
                            three = np.random.choice(
                                np.setdiff1d(
                                    np.arange(
                                        2*int(np.prod(self.dim)/3),
                                        int(np.prod(
                                            self.dim))),
                                    all_indeces[mc, :]), num_sen_sec_3,
                                replace=False)
                            all_indeces[
                                :, (prev_num_sen
                                     + 2*int(np.prod(self.dim)/3)):] = np.tile(
                                         three, [self.n_MC, 1])
                            sen_idx.append(np.concatenate(
                                [one, two, three]))
            self.sen_idx = all_indeces
            self.grouped_sen_idx = sen_idx
            self.update_gp_observation_sizes()

    def quantize_p_and_z(self, q_borders, q_centers):
        """Quantize p-values and z-scores

        Parameters
        ----------
        q_borders : numpy array
            The edges of the quantization intervals.
        q_centers : numpy array
            The centers of the quantization intervals.
        """
        quan_idx = np.zeros(self.p.shape, dtype=int)
        quan_vls = np.zeros(self.p.shape)
        for mc in np.arange(self.p.shape[0]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in divide")
                quan_idx_mc = np.nanargmin(
                    (self.p[mc, :] / np.tile(q_borders[:, np.newaxis],
                                             [self.p[mc, :].size])) - 1 > 0,
                    0) - 1
            quan_vls_mc = q_centers[quan_idx_mc]
            quan_idx[mc, :] = quan_idx_mc
            quan_vls[mc, :] = quan_vls_mc
        self.p = quan_vls
        self.quan_idx = quan_idx
        self.z = stats.norm.ppf(self.p)

    def sha_fa(self, x_crd, y_crd, x_cen, y_cen, sig_s):        
        """
            Simulate shadow fading for a given coordinate vectors in a 2D
            field and a single transmitter placed at a given location.

            Parameters
            ----------
            x_crd : array
                The coordinate vector in x-domain.
            y_crd : array
                The coordinate vector in y-domain.
            x_cen : float
                The transmitter location x-coordinate.
            y_cen : float
                The coordinate vector in y-domain.
            sig_s: float
                The shadow fading variance. As its value indicates whether
                urban or suburban area is to be simulated, the correlation
                distance is chosen accordingly in this method.

            Returns
            -------
            None.

        @author: Martin Goelz
        """

        #The inverse correlation distance [Biglieri2012] -> the longer, the
        #stronger the correlation over distance
        if sig_s==4:#urban (according to [Cai2003])
            a = self.gp_lng*0.1204
        elif sig_s==8:
            a = self.gp_lng*1/500 #suburban
        else:
            print("No valid shadow fading standard deviation selected!")

        #Sum-of-sinusoids (SOS) parameters
        M = 100 #The number of different radial frequencies. The larger, the
        #more accurate.

        #Setup of SOS
        N = 2*M**2 #The number of sampling points of the spatial
        #auto-correlation fct. Equivalent to the total number of frequencies
        #in x and y domain.
        n = np.arange(1, N+1, 1) #Frequency index vector

        eta = np.sqrt(2/N)+np.zeros(N) #Pre-factors for the sinusoids
        tht = stats.uniform(scale=2*np.pi).rvs(N) #The random phase offset

        #f_c_sf = 0.19#Cut-off frequency for the correlation function [Cai2003]
        f_c_sf = a*(10**(2*30/30)-1)/(2*np.pi*self.gp_lng)#30-dB cut-off
        #frequency for the acf [Cai2003]
        P = 1-(a/np.sqrt(a**2+4*np.pi**2*f_c_sf**2))#Mean total power between
        #0 & cut-off
        f_r = np.zeros(M+1)#Initialization of radial frequency vector

        #Sampling the radial domain such that between each sampled radial
        #frequencypair, an equal amount of power is contained. This is the
        #nonuniform sampling method [Cai2003].
        for idx in np.arange(1, M+1):
            f_r[idx] = 1/(2*np.pi)*np.sqrt(
                (P/((M)*a) - 1/np.sqrt(a**2+4*np.pi**2*f_r[idx-1]**2))**(-2)
                - a**2)
            #The formula used here is different from Eq. (17) in [Cai2003],
            #which appears to be wrong. Re-derivation of the equation leads to
            #the implementation here.

        #Obtaining the equally-spaced frequency angles (refer to Fig. 2,
        #[Cai2003])
        i = np.arange(0, 2*M)
        var_phi = (np.pi)*(2*i-2*M+1)/(4*M)

        #The spatial correlation frequencies
        f_x = np.zeros(N)
        f_y = np.zeros(N)

        nu = np.floor_divide(n-1, 2*M)
        for idx in np.arange(0, N, 1):
            k = nu[idx] +1
            l = n[idx]-2*nu[idx]*M-1
            f_x[idx] = f_r[k]*np.cos(var_phi[l])
            f_y[idx] = f_r[k]*np.sin(var_phi[l])

        #Generation of the simulated shadow-fading map
        s = np.zeros((y_crd.size, x_crd.size))
        dis_gp_cen_X = x_crd-x_cen
        dis_gp_cen_Y = y_crd-y_cen
        # s = np.dot(
        #     eta, np.cos(2*np.pi*(f_x*dis_gp_cen_X + f_y*dis_gp_cen_Y) + tht))
        for x_idx in np.arange(0, x_crd.size, 1):
            for y_idx in np.arange(0, y_crd.size, 1):
                s[y_idx, x_idx] = np.dot(eta, np.cos(
                    2*np.pi*(f_x*dis_gp_cen_X[x_idx]
                             + f_y*dis_gp_cen_Y[y_idx]) + tht))
        return s

    def res_fd(self):
        """
        Reset the field to the null state.

        Returns
        -------
        None.

        @author: Martin Goelz
        """

        self.upd_nul_pro(1)
        self.n_src = 0
        self.A = []
        self.Q = []
        [gp.rad_fd_ini() for gp in self.grd_pt]
        self.upd_tst()
        self.r_tru = np.zeros((self.n_MC, self.n))

    def update_gp_observation_sizes(self):
        """Update the number of observations at all grid points.
        """
        for gp_idx in np.arange(self.n):
            gp_num_obs = np.tile(self.n_obs_sam, [self.n_MC])
            for sen_type in np.arange(self.n_sen.size):
                gp_num_obs[
                    np.where(self.grouped_sen_idx[sen_type]==gp_idx)[0]] = (
                        self.n_obs_per_sen[sen_type])
            self.grd_pt[gp_idx].sel_subset_of_obs(gp_num_obs)
            print(f'\rUpdated num of observations at gp: {gp_idx+1}/{self.n}',
            end="")
        self.upd_tst()

    def upd_tst(self):
        """Update the field with the test statistic values from the grid points
        """
        self.tau = np.array([gp.tau for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()
        self.p = np.array([gp.p for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()
        self.z = np.array([gp.z for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()
        self.n_obs_per_sen = np.array([gp.n_obs_sam_used for gp in
                           self.grd_pt]).reshape(self.n, self.n_MC).transpose()

class RadioSpatialFieldEstimated(object):
    """
    The objects of this class represent an estimated spatial field.

    The objects of this class summarize what the fusion center
    knows about the field based on sensor measurements. Each object of this
    class hence has a corresponding object of class RadioSpatialField that it
    tries to approximate in terms of the field values as closely as possible.

    @author: Martin Goelz
    """

    def __init__(self, fd):
        if isinstance(fd, RadioSpatialField):
            self.n_MC = fd.n_MC
            self.dim = fd.dim
            self.gp_lng = fd.gp_lng
            self.n_gp = fd.n
            self.n = fd.tot_n_sen
            self.scen = fd.scen + '_est'
            self.n_0_gp_hat = np.nan
            self.n_1_gp_hat = np.nan
            self.pi0_gp_hat = np.nan
            self.p = np.nan + np.zeros((self.n_MC, self.n))
            self.z = np.nan + np.zeros((self.n_MC, self.n))
            self.tau = np.nan + np.zeros((self.n_MC, self.n))
            self.p_gp = np.nan + np.zeros((self.n_MC, self.n_gp))
            self.z_gp = np.nan + np.zeros((self.n_MC, self.n_gp))
            self.tau_gp = np.nan + np.zeros((self.n_MC, self.n_gp))
            self.sen_idx = fd.sen_idx
            self.r_tru = np.take_along_axis(
                fd.r_tru, fd.sen_idx, axis=1)
            self.n_1 = np.sum(self.r_tru, axis=1)
            self.pi0 = (self.n-self.n_1)/self.n
            self.ipl_idx = np.zeros((self.n_MC, self.n_gp))  # Indicators
            # whether an
            # interpolated value exists at this grid point. Needed for
            # incomplete interpolations as DT.
            self.n_obs_sam = fd.n_obs_sam  # The maximal number of observations
            # available at the sensors
            self.n_obs_sen = np.array(
                [x.shape[1] for x in fd.grouped_sen_idx])  # Number of
            # observation
            self.sen_m_obs = fd.n
            self.Tau_0 = stats.chi2(self.n_obs_sam)
            self.grd_pt = []
            self.gen_grd_pt()

            self.A = fd.A
            self.Q = fd.Q
            self.X = np.take_along_axis(fd.X, self.sen_idx, axis=1)

            self.n_obs_per_sen = np.zeros((self.n_MC, self.n))
            self.sen_grd_pt = []
            for idx in np.arange(0, self.n_MC, 1):
                self.p[idx, :] = fd.p[
                    idx, self.sen_idx[idx, :]]
                self.p_gp[idx, self.sen_idx[idx, :]] = fd.p[
                    idx, self.sen_idx[idx, :]]
                self.tau[idx, :] = fd.tau[
                    idx, self.sen_idx[idx, :]]
                self.tau_gp[idx, self.sen_idx[idx, :]] = fd.tau[
                    idx, self.sen_idx[idx, :]]
                self.z_gp[idx, self.sen_idx[idx, :]] = fd.z[
                    idx, self.sen_idx[idx, :]]
                self.z[idx, :] = fd.z[
                    idx, self.sen_idx[idx, :]]
                self.n_obs_per_sen[idx, :] = fd.n_obs_per_sen[
                    idx, self.sen_idx[idx, :]]
            # Give the measurements down to the grid points
            self.upd_grd_pt()

            self.sen_cds = np.zeros((self.n_MC, self.sen_idx.shape[1], 2))
            for idx in np.arange(0, self.n_MC, 1):
                for idx2 in np.arange(0, self.sen_idx.shape[1], 1):
                    self.sen_cds[idx, idx2, :] = np.array(
                        [self.grd_pt[self.sen_idx[idx, idx2]].crd[1],
                        self.grd_pt[self.sen_idx[idx, idx2]].crd[0]])
        else:
            print("Requires an object of RadioSpatialField!")

    def crd_to_sgl_id_2D(self, crd):
        """
        Compute the singular index of the given crd w.r.t the spatial field.

        Parameters
        ----------
        crd : tuple
            The 2D coordinates. Either a scalar tuple or a tuple of arrays.

        Returns
        -------
        int
            The singular index of this coordinate.

        @author: Martin Goelz
        """
        return self.dim[0]*crd[0] + crd[1]

    def gen_grd_pt(self):
        """
        Generate grid points for this estimated spatial field.

        Returns
        -------
        None.

        @author: Martin Goelz
        """
        sgl_id = 0

        x_rge = np.arange(0, self.dim[0], 1)
        y_rge = np.arange(0, self.dim[1], 1)
        sgl_id = 0
        for y in y_rge:
            for x in x_rge:
                self.grd_pt.append(GridPointEstimated(
                    sgl_id, ((y, x)), self.n_MC, np.nan + np.zeros(self.n_MC),
                    np.nan + np.zeros(self.n_MC), np.nan + np.zeros(self.n_MC),
                    np.nan + np.zeros(self.n_MC)))
                print(f'\rGenerated grid point: {sgl_id+1}/{self.n_gp}',
                end="")
                sgl_id += 1
        print('\rGeneration of estimated spatial field completed!')

    def quantize_p_and_z(self, q_borders, q_centers):
        """Quantize p-values and z-scores

        Parameters
        ----------
        q_borders : numpy array
            The edges of the quantization intervals.
        q_centers : numpy array
            The centers of the quantization intervals.
        """
        quan_idx = np.zeros(self.p.shape, dtype=int)
        quan_vls = np.zeros(self.p.shape)
        for mc in np.arange(self.p.shape[0]):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in divide")
                quan_idx_mc = np.nanargmin(
                    (self.p[mc, :] / np.tile(q_borders[:, np.newaxis],
                                             [self.p[mc, :].size])) - 1 > 0,
                    0) - 1
            quan_vls_mc = q_centers[quan_idx_mc]
            quan_idx[mc, :] = quan_idx_mc
            quan_vls[mc, :] = quan_vls_mc
            quan_vls[mc, np.isnan(self.p[mc, :])] = np.nan
        self.p = quan_vls
        self.quan_idx = quan_idx
        self.z = stats.norm.ppf(self.p)

    def sgl_id_to_crd_2D(self, sgl_id):
        """
        Compute the 2D crd of a given singular index w.r.t the spatial field.

        Parameters
        ----------
        sgl_id : array
            The singular indeces to convert. Either scalar or 1D-array

        Returns
        -------
        tuple
            The corresponding 2D coordinates. Either a scalar or array tuple.

        @author: Martin Goelz
        """
        return (np.array((sgl_id/self.dim[0])).astype(int),
                np.remainder(sgl_id, self.dim[0]))

    def upd_grd_pt(self):
        """
        Update the grid points of the field.

        Copying the variables that were updated on the field level to the
        grid points.

        Returns
        -------
        None.

        @author: Martin Goelz
        """
        for gp in self.grd_pt:
            gp.p = self.p_gp[:, gp.sgl_id]
            gp.z = self.p_gp[:, gp.sgl_id]
            gp.tau = self.tau_gp[:, gp.sgl_id]
            gp.sen_idc = ~np.isnan(self.p_gp[:, gp.sgl_id])

class GridPoint(object):
    """
    The objects of this class represent grid points somewhere within the
    scope of the spatial field.

    @author: Martin Goelz
    """

    def __init__(self, sgl_id, crd, h, n_MC, n_tr_sam, n_obs_sam):
        self.sgl_id = sgl_id  # The singular index of this grid point
        self.crd = crd
        self.n_MC = n_MC
        self.h = h  # The true hypotheses during different MC runs
        self.n_tr_sam = n_tr_sam
        self.n_obs_sam = n_obs_sam
        self.n_obs_sam_used = n_obs_sam + np.zeros(self.n_MC).astype(int)

        self.x_obs = np.zeros((self.n_MC, self.n_obs_sam))  # Initialization of
        # observations
        self.p = [] #Initialization of p-values
        self.z = [] #Initialization of z-scores

        self.X_0 = stats.norm(0, 1) # Observations under H0 zero-mean Gaussian
        self.Tau_0 = stats.chi2(self.n_obs_sam)  # Test stat under H0 chi2
        # (for energy detector)

        #Generation of training sample (used to estimate density under H0)
        self.x_tr = self.X_0.rvs((self.n_MC, self.n_tr_sam))

        self.X_1 = stats.norm  # Observations under H1 are Gaussian with
        # flexible parameters.
        self.hyp_tes()

        # Initialize the parameters of the observations and test statistic
        # under the alternative
        self.X_par = None
        self.Tau_par = None

        #The index of the influencing alternative event.
        self.A = 0  # If 0, then no influence by an alternative event.

    def hyp_tes(self):
        """Compute the test statistic and the corresponding p-values
        using the energy detector.
        """
        p_lim = 1e-100  # if p-values are 0 (due to numerical problems), set
        # them to this small non-zero value.
        self.tau = np.zeros(self.n_MC)  # Initialize the test statistics
        for mc in np.arange(self.n_MC):
            self.tau[mc] = np.sum(
                np.abs(self.x_obs[mc, 0:self.n_obs_sam_used[mc]])**2)
        self.p = 1 - stats.chi2.cdf(self.tau, self.n_obs_sam_used)

        # Compensate numerical instabilities for very small values
        self.p[np.where(self.p == 0)[0]] = p_lim
        self.z = stats.norm.ppf(self.p)
        self.comp_for_num_inac(p_lim)

    def comp_for_num_inac(self, p_lim, z_lim=-50):
        """Compensate for numerical inaccuracies in stats.chi2.cdf.

        The function stats.chi2.cdf provides very inaccurate values when large
        test statistics are passed to it. Hence, we use numerical integration
        since it is particularly important to use accurate values in the right
        tail of the null distribution.
        """
        # for server.
        def find_p_val(tau, df, grd_pts=1000):
            grd_max_val = np.arange(df/1e1, df*1e3, df/1e1)
            max_val = grd_max_val[
                np.where(stats.chi2.pdf(grd_max_val, df) == 0)[0][0]]
            if max_val == grd_max_val[-1]:
                print('p-val is being underestimated!')
            grd = np.arange(tau, max_val, (max_val-tau)/grd_pts)
            return np.sum((grd[1:]-grd[0:-1])/2*(stats.chi2.pdf(grd[0:-1], df)
                                                + stats.chi2.pdf(grd[1:], df)))

        z_lim = -50
        p_lim = 1e-100
        upd_idx = np.where(self.p == p_lim)[0]
        if not upd_idx.size == 0:
            for it in np.arange(upd_idx.size):
                self.p[upd_idx[it]] = find_p_val(
                    self.tau[upd_idx[it]], self.n_obs_sam_used[upd_idx[it]])

        self.z[upd_idx] = stats.norm.ppf(self.p[upd_idx])
        self.z[self.z == -np.inf] = z_lim
        self.p[self.z == z_lim] = 1e-305


    def rad_fd_ini(self):
        """Initialize the spatial radio field by noise-only observations.
        """
        self.x_obs = self.X_0.rvs(size=(self.n_MC, self.n_obs_sam))
        self.h = np.zeros(self.n_MC)
        self.Tau_0 = stats.chi2(self.n_obs_sam)
        self.hyp_tes()

    def rad_fd_up_val(self, tra_sig):
        """
        Update the spatial radio field with the given transmit signal values
        at this grid point. Composes the field measurements by adding the
        noise. Also updates the true hypothesis accordingly. Perform also the
        energy-detector hypothesis test.

        Parameters
        ----------
        tra_sig : array
            The transmit signal observed at this grid point.

        Returns
        -------
        None.

        """
        self.x_obs = tra_sig[:, np.newaxis] + self.X_0.rvs(
            size=(self.n_MC, self.n_obs_sam))
        self.h[np.where(tra_sig != 0)] = np.ones(
            np.sum(tra_sig != 0))
        self.hyp_tes()

    def re_obs(self, n_MC=-1, n_obs_sam=-1, h=np.array([]),
                  X_par=np.array([]), Tau_par=np.array([])):
        """
        Reobserve at the this grid point.

        Parameters
        ----------
        n_MC : int, optional
            The number of MC re-samples. If -1, use the same number of MC
            resamples as before. The default is -1.
        n_obs_sam : int, optional
            The number of observations in the observation sample. If -1, use
            the same number of observations as before. The default is -1.
        h : array, optional
            Indicating the hypothesis during this observation. If -1, use the
            same hypothesis as before. The default is -1.
        X_par : list, optional
            The vector of alternative random variable parameters over MC runs
            at which the alternative is true. If -1, the random variables the
            default value for the alternative RV is used at all alternative MC
            runs. The default is -1.
        Tau_par : list, optional
            The vector of alternative random variable parameters for the test
            statistic over MC runs at which the alternative is true. If -1,
            the random variables the default value for the alternative RV is
            used at all alternative MC runs. The default is -1.

        @author: Martin Goelz
        """

        if n_MC != -1:
            self.n_MC = n_MC
        if n_obs_sam != -1:
            self.n_obs_sam = n_obs_sam
        if h.size != 0:
            self.h = h
        self.Tau_par = Tau_par
        self.X_par = X_par

        self.x_obs = np.zeros((self.n_MC, self.n_obs_sam))
        self.x_obs[np.where(self.h == 0)[0], :] = self.X_0.rvs(
            (np.sum(self.h == 0), self.n_obs_sam))
        self.x_obs[np.where(self.h == 1)[0], :] = self.X_1.rvs(
            self.X_par[:,np.newaxis], 1+np.zeros(
                np.sum(self.h == 1))[:,np.newaxis], size=(
                    (np.sum(self.h==1), self.n_obs_sam)))

    def re_tr(self, n_MC=-1, n_tr_sam=-1):
        """
        Retrain this grid point.

        Parameters
        ----------
        n_MC : int, optional
            The number of MC re-samples. If -1, use the same number of MC
            resamples as before. The default is -1.
        n_tr_sam : int, optional
            The number of observations in the training sample. If -1, use the
            same number of observations as before. The default is -1.

        @author: Martin Goelz
        """

        if n_MC != -1:
            self.n_MC = n_MC
        if n_tr_sam != -1:
            self.n_tr_sam = n_tr_sam
        self.x_tr = self.X_0.rvs((self.n_MC, self.n_tr_sam))

    def sel_subset_of_obs(self, n_obs_sam_used):
        """Update the measures of this grid point such that only a subset
        of observations is used. The number of observations used might vary
        per MC run.

        Parameters
        ----------
        n_obs_sam_used : numpy array
            The number of samples to be used in each MC run.
        """
        self.n_obs_sam_used = n_obs_sam_used
        self.hyp_tes()

    @property
    def crd(self):
        """
        The location of the grid point.
        """
        return self._crd

    @property
    def h(self):
        """
        The true hypethesis.
        """
        return self._h

    @property
    def n_MC(self):
        """
        The number of Monte Carlo runs. Has to be a positive integer.
        """
        return self._n_MC

    @crd.setter
    def crd(self, vls):
        idc = True
        if ((not isinstance(vls, tuple))
            and np.issubdtype(type(vls), np.integer)):
            self._crd = vls
        else:
            for vl in vls:
                if not np.issubdtype(type(vl), np.integer):
                    idc = False
            if idc:
                self._crd = vls
            else:
                print('Coordinates have different dimensions')

    @h.setter
    def h(self, vls):
        if np.array(vls).size != self.n_MC:
            print('Less hypotheses than MC runs specified!')
        if np.all(np.logical_or(vls == 0, vls == 1)) or np.all(np.isnan(vls)):
            self._h = vls
        else:
            print('At least one invalid Hypothesis!')

    @n_MC.setter
    def n_MC(self, vl):
        if not isinstance(vl, int):
            print('Was given non-int number of MC runs!')
        if vl > 0:
            self._n_MC = int(vl)
        else:
            print('Hand over a positive number of MC runs!')

    @crd.deleter
    def crd(self):
        del self._crd

    @h.deleter
    def h(self):
        del self._h

    @n_MC.deleter
    def n_MC(self):
        del self._n_MC

class GridPointEstimated(object):
    """
    The objects of this class represent grid points somewhere within the
    scope of the estimated spatial field. Represents what we know at the
    fusion center about the single grid points.

    @author: Martin Goelz
    """

    def __init__(self, sgl_id, crd, MC, p, z, tau, sen_idc):
        self.sgl_id = sgl_id #The singular index of this grid point
        self.crd = crd
        self.MC = MC
        self.p = p
        self.z = z
        self.tau = tau
        self.sen_idc = sen_idc
        self.Z_0 = stats.norm

    @property
    def crd(self):
        """
        The location of the grid point.
        """
        return self._crd

    @property
    def MC(self):
        """
        The number of Monte Carlo runs. Has to be a positive integer.
        """
        return self._MC

    @property
    def p(self):
        """
        The observed and estimated p-values at this grid point for all MC
        runs.
        """
        return self._p

    @property
    def z(self):
        """
        The observed and estimated z-scores at this grid point for all MC
        runs.
        """
        return self._z

    @property
    def tau(self):
        """
        The observed and estimated test statistics at this grid point for
        all MC runs.
        """
        return self._tau

    @property
    def sen_idc(self):
        """
        An indicator with entries = 1 if we have measurement data at this
        grid point or = 0, if the quantities here are based on interpolation.
        """
        return self._sen_idc

    @property
    def sgl_id(self):
        """
        The 1D index of this grid point
        """
        return self._sgl_id

    @crd.setter
    def crd(self, vls):
        idc = True
        if ((not isinstance(vls, tuple))
            and np.issubdtype(type(vls), np.integer)):
            self._crd = vls
        else:
            for vl in vls:
                if not np.issubdtype(type(vl), np.integer):
                    idc = False
            if idc:
                self._crd = vls
            else:
                print('Coordinates have different dimensions')

    @MC.setter
    def MC(self, vl):
        if not isinstance(vl, int):
            print('Was given non-int number of MC runs!')
        if vl > 0:
            self._MC = int(vl)
            self.p = np.zeros(self.MC)
            self.tau = np.zeros(self.MC)
            self.sen_idc = np.zeros(self.MC)
        else:
            print('Hand over a positive number of MC runs!')

    @p.setter
    def p(self, vl):
        if vl.size == self.MC:
            self._p = vl
        else:
            print('Given p-values are not of right dimension!')

    @z.setter
    def z(self, vl):
        if vl.size == self.MC:
            self._z = vl
        else:
            print('Given z-scores are not of right dimension!')

    @tau.setter
    def tau(self, vl):
        if vl.size == self.MC:
            self._tau = vl
        else:
            print('Given test statistics are not of right dimension!')

    @sgl_id.setter
    def sgl_id(self, vl):
        self._sgl_id = vl

    @sen_idc.setter
    def sen_idc(self, vl):
        if vl.size == self.MC:
            self._sen_idc = vl
        else:
            print('Given sensor indicators are not of right dimension!')

    @sgl_id.setter
    def sgl_id(self, vl):
        self._sgl_id = vl

    @crd.deleter
    def crd(self):
        del self._crd

    @MC.deleter
    def MC(self):
        del self._MC

    @p.deleter
    def p(self):
        del self._p

    @z.deleter
    def z(self):
        del self._z

    @tau.deleter
    def tau(self):
        del self._tau

    @sen_idc.deleter
    def sen_idc(self):
        del self._sen_idc

    @sgl_id.deleter
    def sgl_id(self):
        del self._sgl_id

# %% functions
def cr_fd(fd_scen, dat_path, kind):
    """Create an instance of RadioSpatialField for the given scenario.

    Parameters
    ----------
    fd_scen : str
        The scenario the field is to be created for.
    dat_path : str
        The path to where the data is stored.

    Returns
    -------
    RadioSpatialField
        The instance of class RadioSpatialField
    """
    print('Field will be created...', end="")
    if kind == 'custom':
        fd = CustomSpatialField(fd_scen, os.path.join(dat_path, '..'))
        ls.sv_fd(os.path.join(dat_path, 'fd'), fd)
    elif kind == 'non-spatial':
        fd = FakeField(fd_scen, os.path.join(dat_path, '..'))
        ls.sv_fd(os.path.join(dat_path, 'fd'), fd)
    else:
        [fd_dim, n_MC, n_sam, n_src, ran_cen, ran_rad, ran_pre, add_tra,
         pi0_des, sha_fa, prop_env] = ls.ld_sc(
             os.path.join(dat_path), fd_scen)
        fd = RadioSpatialField(fd_scen, fd_dim, n_MC, 100, n_sam)
        fd.pla_tra(fd_scen, dat_path)
        ls.sv_fd(dat_path, fd)
    print("Field created and stored successfully!")
    return fd


def rd_in_fds(fd_scen, sen_cfg, dat_path):
    """Read in and return the fields for the given config and scenario.

    Parameters
    ----------
    fd_scen : str
        The scenario name.
    sen_cfg : str
        The configuration name.
    dat_path : str
        The path to where the data is stored.

    Returns
    -------
    RadioSpatialField
        The object of RadioSpatialField.
    RadioSpatialFieldEstimated
        The corresponding object of RadioSpatialFieldEstimated.
    """
    try:
        if not (sen_cfg == 'custom' or sen_cfg == 'non-spatial'):
            fd = ls.ld_fd(os.path.join(dat_path, sen_cfg, fd_scen))
        else:
            fd = ls.ld_fd(os.path.join(dat_path, 'fd'))
        try:
            if not (sen_cfg == 'custom' or sen_cfg == 'non-spatial'):
                est_fd = ls.ld_fd(os.path.join(dat_path, sen_cfg, fd_scen)
                                  + '_est')
            else:
                est_fd = ls.ld_fd(os.path.join(dat_path, 'fd_est'))
        except FileNotFoundError:
            if not (sen_cfg == 'custom' or sen_cfg == 'non-spatial'):
                est_fd = RadioSpatialFieldEstimated(fd)
                ls.sv_fd(os.path.join(dat_path, sen_cfg, fd_scen) + '_est',
                         est_fd)
            else:
                if sen_cfg == 'custom':
                    est_fd = CustomSpatialFieldEstimated(fd)
                else:
                    est_fd = fd
                ls.sv_fd(os.path.join(dat_path, 'fd_est'), est_fd)
    except FileNotFoundError:
        if not (sen_cfg == 'custom' or sen_cfg == 'non-spatial'):
            try:
                fd = ls.ld_fd(dat_path)
                fd.pla_sen(dat_path, sen_cfg)
                ls.sv_fd(os.path.join(dat_path, sen_cfg, fd_scen), fd)
            except FileNotFoundError:
                # Create file
                print('Such a field does not exist yet!')
                fd = cr_fd(fd_scen, dat_path, sen_cfg)
                fd.pla_sen(dat_path, sen_cfg)
                ls.sv_fd(os.path.join(dat_path, sen_cfg, fd_scen), fd)
            est_fd = RadioSpatialFieldEstimated(fd)
            ls.sv_fd(os.path.join(dat_path, sen_cfg, fd_scen) + '_est', est_fd)
        else:
            fd = cr_fd(fd_scen, dat_path, sen_cfg)
            if sen_cfg == 'custom':
                est_fd = CustomSpatialFieldEstimated(fd)
            else:
                est_fd = fd
            ls.sv_fd(os.path.join(dat_path, 'fd_est'), est_fd)
    return fd, est_fd
