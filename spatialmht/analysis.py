#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides plotting capabilities.

@author: Martin Gölz
"""
import numpy as np
import scipy.stats as stats

import field_handling as fd_hdl
import tuda_colors

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from tabulate import tabulate

# The color dictionairy for my linearly spaced color-map emphasizing small
# p-values. Setup such that yellow is exactly at 0.15
color_dic = {'red':   [(0.0,  0.0, 1.0),
                       (0.15, 1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             'green': [(0.0,  0.0, 0.0),
                       (0.15, 1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'blue':  [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}

# Alternative colormap fading from red (small p-values) to white instead of
# green (might be needed for colorblind people).
color_dic_ry = {'red':   [(0.0,  0.0, 1.0),
                          (0.15,  1.0, 1.0),
                          (1.0,  1.0, 0.0)],

                'green':  [(0.0,  0.0, 0.0),
                           (0.15,  1.0, 1.0),
                           (1.0,  1.0, 0.0)],

                'blue':  [(0.0,  0.0, 0.0),
                          (1.0,  1.0, 0.0)]}

# Definition of custom colormaps for visualization of lfdrs.
color_dic_lfdrs = {'red':   [(0.0,  0.0, 1.0),
                       (0.5, 1.0, 1.0),
                       (1.0,  0.0, 0.0)],

             'green': [(0.0,  0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0,  1.0, 1.0)],

             'blue':  [(0.0,  0.0, 0.0),
                       (1.0,  0.0, 0.0)]}

cm_fal_dis = colors.LinearSegmentedColormap.from_list(
    'fal_dis', ['white', tuda_colors.TUDa_9b], 2)
cm_mis_dis = colors.LinearSegmentedColormap.from_list(
    'cor_dis', ['white', '#555555'], 2)
cm_cor_dis = colors.LinearSegmentedColormap.from_list(
    'fal_dis', ['white', tuda_colors.TUDa_4b], 2)
cm_nul_alt = colors.LinearSegmentedColormap.from_list(
    'nulls_and_alternatives', ['white', tuda_colors.TUDa_2b], 2)
cm_dis = colors.LinearSegmentedColormap.from_list(
    'dis', ['white', tuda_colors.TUDa_6d], 2)
 # Can be called under name 'lfdr_cmap'
cm_lfdr = colors.LinearSegmentedColormap('lfdr_cmap', color_dic)


def plot_alt(fd, n_MC=0):
    """
    Plot the associations of pixels with H_1.

    Parameters
    ----------
    fd : SpatialField
        The spatial field to be visualized.
    n_MC : int, optional
        The Monte Carlo run to be visualized. The default is 0.

    Returns
    -------
    None.

    @author: Martin Goelz
    """
    if (isinstance(fd, fd_hdl.RadioSpatialField)
            or isinstance(fd, fd_hdl.CustomSpatialField)):
        # The list of colors, depending on the number of alternative events
        col_lst = tuda_colors.TUDa_cm_lst[0:int(fd.Q.shape[2])]
        col_lst.insert(0, 'white')  # For the cluster of nulls

        # The colormap and the corresponding boundaries
        cm_ev = colors.ListedColormap(col_lst)
        bd_ev = (np.arange(0, 3, 1)-0.01).tolist()

        # The normalization of the colormap
        nor = colors.BoundaryNorm(bd_ev, cm_ev.N, clip=True)

        fig = plt.figure(figsize=(4, 4))
        img = plt.imshow(np.reshape(np.sum(fd.Q[n_MC, :, :], axis=1) > 0,
                                    fd.dim[::-1]), cmap=cm_nul_alt,
                         norm=nor, origin='lower')

        # Getting the plots ready for the paper
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelleft=False,
            labelbottom=False)
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$y$', fontsize=12)
        plt.legend(handles=[mpatches.Patch(color=col_lst[0],
                                           label='$\mathcal{H}_0$'),
                            mpatches.Patch(color=col_lst[1],
                                           label=r"$\mathcal{H}_1$")],
                   prop={'size': 16})
        plt.show()
    elif (isinstance(fd, fd_hdl.RadioSpatialFieldEstimated)
          or isinstance(fd, fd_hdl.CustomSpatialFieldEstimated)):
        col_lst = ['white', tuda_colors.TUDa_cm_lst[0]]  # :int(fd.Q.shape[2])]
        # col_lst.insert(0, 'white') #For the cluster of nulls

        # The colormap and the corresponding boundaries
        cm_ev = colors.ListedColormap(col_lst)
        bd_ev = (np.arange(0, 3, 1)-0.01).tolist()

        # The normalization of the colormap
        nor = colors.BoundaryNorm(bd_ev, cm_ev.N, clip=True)

        fig = plt.figure(figsize=(4, 4))
        img = plt.imshow(np.reshape(fd.r_tru[n_MC, :],
                                    fd.dim[::-1]), cmap=cm_nul_alt,
                         norm=nor, origin='lower')

        # Getting the plots ready for the paper
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            labelleft=False,
            labelbottom=False)
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$y$', fontsize=12)
        plt.legend(handles=[mpatches.Patch(color=col_lst[0],
                                           label='$\mathcal{H}_0$'),
                            mpatches.Patch(color=col_lst[1],
                                           label=r"$\mathcal{H}_1$")],
                   prop={'size': 16})
        plt.show()
    return fig


def plot_ev(fd, n_MC=0):
    """
    Plot the associations of pixels with alternative events.

    Parameters
    ----------
    fd : SpatialField
        The spatial field to be visualized.
    n_MC : int, optional
        The Monte Carlo run to be visualized. The default is 0.

    Returns
    -------
    None.

    @author: Martin Goelz
    """
    # The list of colors, depending on the number of alternative events
    col_lst = tuda_colors.TUDa_cm_lst[0:int(fd.Q.shape[2])]
    col_lst.insert(0, 'white')  # For the cluster of nulls

    # The colormap
    cm_ev = colors.ListedColormap(col_lst)

    # Initialization of colormap to make white of overlaping plots transparent
    cm_ev._init()
    alphas = np.concatenate((np.zeros(1), np.ones(cm_ev.N+2)))
    cm_ev._lut[:, -1] = alphas

    plt.figure()
    ax = plt.gca()

    for k in np.arange(fd.Q.shape[2]):
        dat_mat = (fd.Q[n_MC, :, k] == 1).astype(int)
        dat_mat[np.where(dat_mat == 1)] = k+1
        plt.imshow(np.reshape(dat_mat, fd.dim[::-1]), cmap=cm_ev, vmin=0,
                   alpha=0.85, vmax=int(fd.Q.shape[2]), origin='lower')

    plt.title('The event associations for n_MC run {mc}'.format(
        mc=n_MC))
    ax.set_ylabel('$y$')
    ax.set_xlabel('$x$')
    plt.show()

def plot_fdrs_and_pow(alp_vec, methods, ls_dic, col_dic, label_dic,
                      fig_size=(7.5, 3.5)):
    plt.figure(figsize=fig_size)
    # FDRs
    plt.subplot(1, 2, 1)
    plt.plot(alp_vec, alp_vec, label=r'$\alpha$', c='red')
    for (idx, current_method_res) in enumerate(methods):
        plt.plot(alp_vec, [x.fdr for x in current_method_res], ls_dic[idx],
                 label=label_dic[idx], c=col_dic[idx], lw=1)
    plt.legend()
    plt.xlabel('nominal FDR')
    plt.ylabel('empirical FDR')
    plt.title('FDR')
    # Power
    plt.subplot(1, 2, 2)
    for (idx, current_method_res) in enumerate(methods):
        plt.plot(alp_vec, [x.pow for x in current_method_res], ls_dic[idx],
                 label=label_dic[idx], c=col_dic[idx], lw=1)
    plt.legend()
    plt.xlabel('nominal FDR')
    plt.ylabel('detection power')
    plt.title('Power')

def plot_lfdrs(lfdr, fd, mc, interpolation=None, show_cb=True,
               fig_size=(4,4)):
    """
    Visualize lfdrs in a spatial grid.

    Parameters
    ----------
    lfdr : float vector
        The lfdrs to be visualized.
    fd: RadioSpatialField or RadioSpatialFieldEstimated
        The field
    interpolation: str
        The interpolation strategy for imshow (Only for visualization!)
    show_cb: bool
        If true, show colorbar.
    Returns
    -------
    None.
)
    @author: Martin Goelz
    """
    fig, ax = plt.subplots(figsize=fig_size)
    if not isinstance(fd, fd_hdl.FakeField):
        if (isinstance(fd, fd_hdl.RadioSpatialField)
                or isinstance(fd, fd_hdl.CustomSpatialField)):
            hm = plt.imshow(lfdr[mc, :].reshape(fd.dim),
                            extent=[0, fd.dim[1], 0, fd.dim[0]], origin='lower',
                            cmap=cm_lfdr, interpolation=interpolation)
            if show_cb:
                fig.colorbar(hm, fraction=0.046, pad=0.04)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
    
            show_sensors_in_field(
                fd.sgl_id_to_crd_2D(fd.sen_idx[mc, :]),
                color='black', linewidth=1.5)
        elif (isinstance(fd, fd_hdl.RadioSpatialFieldEstimated)
              or isinstance(fd, fd_hdl.CustomSpatialFieldEstimated)):
            dat = np.zeros(np.prod(fd.dim)) + np.nan
            dat[fd.sen_idx[mc, :]] = (lfdr[mc, :])
    
            hm = plt.imshow(dat.reshape(fd.dim[::-1]), cmap=cm_lfdr,
                            extent=[0, fd.dim[1], 0, fd.dim[0]], origin='lower')
    
            show_sensors_in_field(
                fd.sgl_id_to_crd_2D(fd.sen_idx[mc, :]),
                color='black', linewidth=1.5)
            if show_cb:
                fig.colorbar(hm, ax=ax, ticks=list(np.linspace(0, 1, 11)))
            plt.title(f'The $p$-values for MC run {mc}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xlabel('$c_x$')
        plt.xlabel('$x$')
        # plt.ylabel('$c_y$')
        plt.ylabel('$y$')
    else:
        hm = plt.scatter(np.arange(lfdr.shape[1]) + 1, lfdr[mc, :], marker='x')
        ax.axes.get_xaxis().set_ticks([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Hypothesis number $n$')
        plt.ylabel('lfdr')
    plt.title('The lfdrs')
    return fig

def plot_rej(det_res, fd, n_MC=0):
    """
    Plot the detection results, where correct detections turn out green,
    missed ones grey and false ones red.

    Parameters
    ----------
    det_res : DetectionResult
        The detection result to be visualized.
    fd : SpatialField
        The spatial field to be visualized.
    n_MC : int, optional
        The Monte Carlo run to be visualized. The default is 0.

    Returns
    -------
    None.

    @author: Martin Goelz
    """

    if det_res.sen == False:
        plot_rej_su(det_res, fd, n_MC=n_MC)
    else:
        plot_rej_sen(det_res, fd, n_MC=n_MC)


def plot_rej_su(det_res, fd, n_MC=0, fig_size=(4, 4)):
    """
    Plot the detection results on the spatial unit level.

    Parameters
    ----------
    det_res : DetectionResult
        The detection result to be visualized.
    fd : SpatialField
        The spatial field to be visualized.
    n_MC : int, optional
        The Monte Carlo run to be visualized. The default is 0.

    Returns
    -------
    None.

    @author: Martin Goelz
    """
    
    if isinstance(fd, fd_hdl.FakeField):
        fix, ax = plt.subplots(figsize=fig_size)
        px_rge = np.arange(0, fd.n)
        p = fd.p[n_MC, :]
        if np.sum(np.isnan(det_res.r_tru)) == np.prod(fd.p.shape):
            dat = (det_res.r_det[n_MC, :] == 0, det_res.r_det[n_MC, :] == 1)
            col_mp = ['white', tuda_colors.TUDa_6d]
            nam_mp = ['Non-Discovery', 'Discovery']
        else:
            dat = (det_res.u[n_MC, :], det_res.s[n_MC, :], det_res.v[n_MC, :],
                   det_res.t[n_MC, :])
            col_mp = ['white', tuda_colors.TUDa_4b, tuda_colors.TUDa_9b,
                      '#555555']
            nam_mp = ['Correct Non-Discovery', 'Correct Discovery',
                      'False Discovery', 'Missed Discovery']
        for val, col, nam in zip(dat, col_mp, nam_mp):
            ax.scatter(px_rge[val == 1], p[val == 1],
                       c=col, edgecolors='black',
                       label=nam)
        plt.legend()
        plt.title('detection results')
        ax.set_ylabel('$p$-value')
        ax.set_xlabel('$n$: spatial unit index')
        plt.show()
    elif np.array(fd.dim).shape == ():
        fix, ax = plt.subplots(figsize=fig_size)
        px_rge = np.arange(0, fd.dim)
        p = fd.p[n_MC, :]
        dat = (det_res.u[n_MC, :], det_res.s[n_MC, :], det_res.v[n_MC, :],
               det_res.t[n_MC, :])
        col_mp = ['white', tuda_colors.TUDa_4b, tuda_colors.TUDa_9b,
                  '#555555']
        nam_mp = ['Correct Non-Discovery', 'Correct Discovery',
                  'False Discovery', 'Missed Discovery']
        for val, col, nam in zip(dat, col_mp, nam_mp):
            ax.scatter(px_rge[val == 1], p[val == 1],
                       c=col, edgecolors='black',
                       label=nam)
        plt.legend()
        plt.title('detection results')
        ax.set_ylabel('$p$-value')
        ax.set_xlabel('$n$: spatial unit index')
        plt.show()
    else:
        # Initialization of colormap to make white of overlaying plot
        # transparent
        cm_fal_dis._init()
        cm_mis_dis._init()
        cm_dis._init()
        cm_fal_dis._lut[:, -1] = np.array([0, 1, 1, 1, 1])
        cm_mis_dis._lut[:, -1] = np.array([0, 1, 1, 1, 1])
        cm_dis._lut[:, -1] = np.array([0, 1, 1, 1, 1])

        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        if np.sum(np.isnan(det_res.r_tru)) == np.prod(fd.p.shape):
            plt.imshow(np.reshape(det_res.r_det[n_MC, :], fd.dim[::-1]),
                       cmap=cm_dis, vmin=0-.5, vmax=1+.5, origin='lower')
        else:
            plt.imshow(np.reshape(det_res.s[n_MC, :], fd.dim[::-1]),
                       cmap=cm_cor_dis, vmin=0-.5, vmax=1+.5, origin='lower')
            plt.imshow(np.reshape(det_res.v[n_MC, :], fd.dim[::-1]),
                       cmap=cm_fal_dis, vmin=0-.5, vmax=1+.5, origin='lower')
            plt.imshow(np.reshape(det_res.t[n_MC, :], fd.dim[::-1]),
                       cmap=cm_mis_dis, vmin=0-.5, vmax=1+.5, origin='lower')
        plt.title('detection results')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_ylabel('$c_y$')
        # ax.set_xlabel('$c_x$')
        ax.set_ylabel('$y$')
        ax.set_xlabel('$x$')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        plt.show()


def plot_rej_sen(det_res, fd, n_MC=0, fig_size=(4, 4)):
    """
    Plot the detection results on the sensor level.

    Parameters
    ----------
    det_res : DetectionResult
        The detection result to be visualized.
    fd : SpatialField
        The spatial field to be visualized.
    n_MC : int, optional
        The Monte Carlo run to be visualized. The default is 0.

    Returns
    -------
    None.

    @author: Martin Goelz
    """
    if (isinstance(fd, fd_hdl.RadioSpatialField)
            or isinstance(fd, fd_hdl.CustomSpatialField)):
        plot_alt(fd, n_MC=n_MC)
    else:
        fig, ax = plt.subplots(figsize=fig_size)
    ax = plt.gca()
    px_rge = np.arange(0, np.prod(fd.dim))[fd.sen_idx[n_MC, :]]
    if np.sum(np.isnan(det_res.r_tru)) == np.prod(fd.p.shape):
        dat = (det_res.r_det[n_MC, :] == 0, det_res.r_det[n_MC, :] == 1)
        col_mp = ['white', tuda_colors.TUDa_6d]
        nam_mp = ['Non-Discovery', 'Discovery']
    else:
        dat = (det_res.u[n_MC, :], det_res.s[n_MC, :],
               det_res.v[n_MC, :],
               det_res.t[n_MC, :])
        col_mp = ['white', tuda_colors.TUDa_4b, tuda_colors.TUDa_9b,
                  '#555555']
        nam_mp = ['Correct Non-Discovery', 'Correct Discovery',
                  'False Discovery', 'Missed Discovery']
    if np.array(fd.dim).shape == ():
        p = fd.p[n_MC, fd.sen_idx[n_MC, :]]
        for val, col, nam in zip(dat, col_mp, nam_mp):
            ax.scatter(px_rge[val == 1], p[val == 1], marker='s',
                       c=col, edgecolors='black', linewidth=1.5,
                       label=nam)
    else:
        for val, col, nam in zip(dat, col_mp, nam_mp):
            ax.scatter(fd.sgl_id_to_crd_2D(px_rge[val == 1])[1],
                       fd.sgl_id_to_crd_2D(px_rge[val == 1])[0],
                       marker='s', c=col, edgecolors='black', linewidth=1.5,
                       label=nam)
    plt.legend()
    plt.title('detection results')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not np.array(fd.dim).shape == ():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('$y$')
        ax.set_xlabel('$x$')
    else:
        ax.set_ylabel('$p$-value')
        ax.set_xlabel('$n$: spatial unit index')
    plt.xlim([0-.5, fd.dim[1]-.5])
    plt.ylim([0-.5, fd.dim[0]-.5])
    plt.show()


def plot_pvals(fd, n_MC=0, sen_only=False):
    """
    If the spatial field is 1D, plot a scatter plot of the p-values. If
    the spatial field is 2D or 3D, plot a heatmap of the p-values.

    Parameters
    ----------
    fd: SpatialField
        The spatial field to be visualized
    n_MC : float, optional
        The Monte Carlo run the field is to be visualized for.
        The default is 0.
    sen_only : boolean
        If true, only sensor measurements visualized

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure this heatmap is plotted in.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis this heatmap is plotted in.
    cb : matplotlib.colorbar.Colorbar
        The colorbar. If the field is 1D, this is None.

    @author: Martin Goelz
    """
    if isinstance(fd, fd_hdl.FakeField):
        return (plot_pvals_1D(fd, n_MC), None)
    if isinstance(fd.dim, tuple) and 0 <= n_MC < fd.n_MC:
        return plot_pvals_2D(fd, n_MC=n_MC, sen_only=sen_only)
    elif isinstance(fd.dim, tuple):
        print('Cannot visualize: This is not a valid n_MC run index!')
    else:
        return (plot_pvals_1D(fd, n_MC), None)


def plot_pvals_1D(fd, n_MC=0, id_2D=0, id_3D=0):
    """
    Plot a 1D scatter plot of the p-values for the specified Monte Carlo
    run. If the field is 2D or 3D, the field is "cut" along id_2D in the
    y-plane and along id_3D in the z-plane (if existing).

    Parameters
    ----------
    fd: SpatialField/FakeField
        The spatial field to be visualized
     n_MC : float, optional
        The Monte Carlo run the field is to be visualized for.
        The default is 0.
    id_2D : float, optional
        If the field is 2D or 3D, the y-plane to be cut along. The default
        is 0.
    id_3D : float, optional
        If the field is 3D, the z-plane to be cut along. The default is 0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure this scatter plot is plotted in.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis this scatter plot is plotted in.

    @author: Martin Goelz

    """
    if isinstance(fd, fd_hdl.FakeField) or not isinstance(fd.dim, tuple):
        if 0 <= n_MC < fd.n_MC:
           fig, ax = plt.subplots()
           ax.scatter(np.arange(0, fd.n), fd.p[n_MC, ], label='p-values')
        else:
           print('Cannot visualize: This is not a valid MC run index!')
    elif isinstance(fd.dim, tuple) and 0 <= n_MC < fd.n_MC:
        fig, ax = plt.subplots()
        p_str = fd.p[n_MC, ].reshape(fd.dim[::-1])
        if len(fd.dim) == 3:
            if not 0 <= id_2D < fd.dim[1]:
                print('2D identifier not within valid range! Reset to 0.')
                id_2D = 0
            if not 0 <= id_3D < fd.dim[2]:
                print('3D identifier not within valid range! Reset to 0.')
                id_3D = 0
            ax.scatter(np.arange(0, fd.dim[0]), p_str[id_3D, id_2D, :],
                       label='Along y = \
{y} and z = {z}'.format(y=id_2D, z=id_3D))
        else:
            if not 0 <= id_2D < fd.dim[1]:
                print('2D identifier not within valid range! Reset to 0.')
                id_2D = 0
            ax.scatter(np.arange(0, fd.dim[0]), p_str[id_2D, :],
                       label='Along y = {y}'.format(y=id_2D))
    else:
        print('Cannot visualize: This is not a valid MC run index!')
    plt.title('The $p$-values for MC run {mc}'.format(mc=n_MC))
    ax.set_ylabel('$p$-value')
    ax.set_xlabel('$n$: spatial unit index')
    return (fig, ax)


def plot_pvals_2D(fd, n_MC=0, id_3D=0, sen_only=False):
    """
    Plot a 2D visualization (heatmap) of this spatial field for the
    specified Monte Carlo run. If the field is 1D, the field is not
    plotted. If the field is 3D, the "cut" along id_3D is visualized in 2D.

    Parameters
    ----------
    fd: SpatialField
        The spatial field to be visualized
    n_MC : float, optional
        The Monte Carlo run the field is to be visualized for.
        The default is 0.
    id_3D : float, optional
        If the field is 3D, the z-plane to be cut along. The default is 0.
    sen_only : boolean
        If true, only sensor measurements visualized

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure this heatmap is plotted in.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis this heatmap is plotted in.
    cb : matplotlib.colorbar.Colorbar
        The colorbar

    @author: Martin Goelz
    """
    cmap = colors.LinearSegmentedColormap('my_map', color_dic)

    if (isinstance(fd, fd_hdl.RadioSpatialField)
            or isinstance(fd, fd_hdl.CustomSpatialField)):
        if isinstance(fd.dim, tuple) and 0 <= n_MC < fd.n_MC:
            fig, ax = plt.subplots()
            if sen_only:
                p_str = np.zeros(fd.p[n_MC, :].size)+np.nan
                p_str[fd.sen_idx[n_MC, :]] = (fd.p[n_MC, fd.sen_idx[n_MC, :]])
            else:
                p_str = fd.p[n_MC, ]
            if len(fd.dim) == 3:
                if not 0 <= id_3D < fd.dim[2]:
                    print('3D identifier not within valid range! Reset to 0.')
                    id_3D = 0
                hm = ax.imshow(p_str.reshape(fd.dim[::-1])[id_3D, :, :],
                               cmap=cmap, vmin=0, vmax=1, origin='lower')
            else:
                hm = ax.imshow(p_str.reshape(fd.dim[::-1]), cmap=cmap,
                               vmin=0, vmax=1, extent=[0, fd.dim[0],
                                                       0, fd.dim[1]],
                               origin='lower')
            if sen_only:
                # ax.scatter(fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[1]+.5,
                #     fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[0]+.5,
                #     c=p_str[fd.sen_idx[n_MC,:]],
                #     marker = 's', edgecolors='black', linewidth=1.5,
                #     cmap=cmap, label='sensor measurements')
                show_sensors_in_field(
                    fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :]),
                    color='black', linewidth=1.5)

            cb = fig.colorbar(hm, ax=ax, ticks=list(np.linspace(0, 1, 11)))
            plt.title('The $p$-values for MC run {mc}'.format(mc=n_MC))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            return ((fig, ax), cb)
        elif isinstance(fd.dim, tuple):
            print('Cannot visualize: This is not a valid MC run index!')
        else:
            print('This field is 1D and cannot be visualized in 2D')
    elif (isinstance(fd, fd_hdl.RadioSpatialFieldEstimated)
          or isinstance(fd, fd_hdl.CustomSpatialFieldEstimated)):
        fig = plt.figure()
        ax = plt.gca()
        if sen_only:
            dat = np.zeros(fd.p[n_MC, :].size)+np.nan
            dat[fd.sen_idx[n_MC, :]] = (fd.p[n_MC, fd.sen_idx[n_MC, :]])
        else:
            dat = (fd.p[n_MC, :])

        hm = plt.imshow(dat.reshape(fd.dim[::-1]), cmap=cmap,
                        extent=[0, fd.dim[1], 0, fd.dim[0]], origin='lower')
        # Adding .5 because the coordinates are always the left lower corner
        # of the pixels
        # ax.scatter(fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[1]+.5,
        #             fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[0]+.5,
        #             c=dat[fd.sen_idx[n_MC,:]],
        #             marker = 's', edgecolors='black', linewidth=1.5,
        #             cmap=cmap, label='sensor measurements')
        show_sensors_in_field(
            fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :]),
            color='black', linewidth=1.5)

        cb = fig.colorbar(hm, ax=ax, ticks=list(np.linspace(0, 1, 11)))
        plt.title(f'The $p$-values for MC run {n_MC}')
        # leg = plt.legend()
        # leg.legendHandles[0].set_facecolor('none')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('$y$')
        ax.set_xlabel('$x$')


def plot_tau(fd, n_MC=0, sen_only=False):
    """
    Plot a 2D visualization (heatmap) of the test statistic for this radio
    patial field for the specified Monte Carlo run.

    Parameters
    ----------
    fd: RadioSpatialField
        The spatial field to be visualized. Currently only working for
        objects of RadioSpatialField, because of proper test statistics
        being implemented only for this class.
    n_MC : float, optional
        The Monte Carlo run the field is to be visualized for.
        The default is 0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure this heatmap is plotted in.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis this heatmap is plotted in.
    cb : matplotlib.colorbar.Colorbar
        The colorbar

    @author: Martin Goelz
    """
    if isinstance(fd, fd_hdl.RadioSpatialField):
        if 0 <= n_MC < fd.n_MC:
            fig, ax = plt.subplots()
            if sen_only:
                dat = np.zeros(fd.tau[n_MC, :].size)+np.nan
                dat[fd.sen_idx[n_MC, :]] = (fd.tau[n_MC, fd.sen_idx[n_MC, :]])
            else:
                dat = fd.tau[n_MC, ]
            hm = ax.imshow(dat.reshape(fd.dim[::-1]), cmap=plt.get_cmap('jet'),
                           norm=colors.LogNorm(vmin=fd.tau[n_MC, ].min(),
                                               vmax=fd.tau[n_MC, ].max()),
                           origin='lower')
            if sen_only:
                sca = ax.scatter(fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[1]+.5,
                                 fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[0]+.5,
                                 c=dat[fd.sen_idx[n_MC, :]],
                                 norm=colors.LogNorm(
                    vmin=fd.tau[n_MC, ].min(),
                    vmax=fd.tau[n_MC, ].max()),
                    marker='s', edgecolors='black', linewidth=1.5,
                    cmap='jet', label='sensor measurements')
            # Polishing the figure
            cb = fig.colorbar(hm, ax=ax)
            plt.title('The energy-detector test statistics for MC run {mc}'.
                      format(mc=n_MC))
            ax.set_ylabel('$y$')
            ax.set_xlabel('$x$')
            return ((fig, ax), cb)
        else:
            print('Cannot visualize: This is not a valid MC run index!')
    elif isinstance(fd, fd_hdl.RadioSpatialFieldEstimated):
        fig, ax = plt.subplots()
        dat = (fd.tau[n_MC, :])

        hm = ax.imshow(dat.reshape(fd.dim[::-1]), cmap='jet',
                       extent=[0, fd.dim[1], 0, fd.dim[0]],
                       norm=colors.LogNorm(
            vmin=dat[np.where(~np.isnan(dat))].min(),
            vmax=dat[np.where(~np.isnan(dat))].max()),
            origin='lower')
        # Adding .5 because the coordinates are always the left lower corner
        # of the pixels
        sca = ax.scatter(fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[1]+.5,
                         fd.sgl_id_to_crd_2D(fd.sen_idx[n_MC, :])[0]+.5,
                         c=dat[fd.sen_idx[n_MC, :]],
                         norm=colors.LogNorm(
                         vmin=dat[np.where(~np.isnan(dat))].min(),
                         vmax=dat[np.where(~np.isnan(dat))].max()),
                         marker='s', edgecolors='black', linewidth=1.5,
                         cmap='jet', label='sensor measurements')
        cb = fig.colorbar(sca, ax=ax)
        plt.title(f'The energy-detector test statistics for MC run {n_MC}')
        ax.set_ylabel('$y$')
        ax.set_xlabel('$x$')
        leg = plt.legend()
        leg.legendHandles[0].set_facecolor('none')

def print_ex_times(ex_times, names):
    srt_idx = np.argsort(np.mean(ex_times, 1))
    pairs = []
    print('--- Execution times sorted from fastest to slowest ---')
    for (idx) in (srt_idx):
        pairs.append([names[idx], int(100000*np.mean(ex_times[idx]))/100000])
    print(
        tabulate(pairs, headers=['Method', 'Average time per run']))
    print('------------------------------------------------------')


def print_fdrs_and_pow(alp_val, res, labels):
    powers = np.array([x.pow for x in res])
    srt_idx = np.argsort(powers)[::-1]

    pairs = []
    print('--- Methods sorted according to detection power ---')

    for idx in np.arange(len(res)):
        if res[srt_idx[idx]].fdr > alp_val:
            suffix = "FDR VIOLATION!!"
        else:
            suffix = ''
        pairs.append([labels[srt_idx[idx]], res[srt_idx[idx]].fdr,
                      res[srt_idx[idx]].pow, suffix])
    print(
        tabulate(pairs, headers=['Method', 'FDR', 'Power', 'FDR Violation?']))

    print('------------------------------------------------------')

def show_sensors_in_field(sen_crds, ax=None, **kwargs):
    for sen in range(sen_crds[0].size):
        rect = plt.Rectangle(
            (sen_crds[1][sen], sen_crds[0][sen]), 1, 1, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
