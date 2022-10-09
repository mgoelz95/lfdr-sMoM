#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides functionalities to load and save files for the package.

@author: Martin Goelz
"""
import os

import _pickle as pickle

def ld_pickle(fil_str):
    """Load a pickle file at the given path.

    Parameters
    ----------
    fil_str : str
        The path to the file.

    Returns
    -------
    pickle
        The loaded pickle object.
    """
    with open(fil_str + '.pkl', 'rb') as input:
        return pickle.load(input)

def ld_cfg(dat_path, sen_cfg):
    """Load a sensor configuration.

    Parameters
    ----------
    dat_path : str
        The path to where the data is stored.
    sen_cfg : str
        The sensor configuration.

    Returns
    -------
    list
        The parameters for this sensor configuration.
    """
    fil_str = os.path.join(dat_path, sen_cfg, sen_cfg) + '_par'
    cfg_par = ld_pickle(fil_str)
    return [cfg_par[x] for x in range(len(cfg_par))]


def ld_fd(fd_path):
    """Load a field.

    Parameters
    ----------
    fd_path : str
        The path to the field.

    Returns
    -------
    RadioSpatialField
        The loaded object of type RadioSpatialField.
    """
    return ld_pickle(fd_path)


def ld_sc(dat_path, fd_scen):
    """Load the parameters of a given field scenario.

    Parameters
    ----------
    dat_path : str
        THe path to where the data is stored.
    fd_scen : str
        The name of the scenario to be loaded.

    Returns
    -------
    list
        The scenario parameters.
    """
    fil_str = os.path.join(dat_path, fd_scen) + '_par'
    fd_par = ld_pickle(fil_str)
    return [fd_par[x] for x in range(len(fd_par))]

def sv_fd(dat_path, fd):
    """Save the given RadioSpatialField in a pickle file under the given name.

    Parameters
    ----------
    dat_path : str
        The path to the storing location.
    fd : RadioSpatialField
        The object to be saved.

    Returns
    -------
    None.

    @author: Martin Goelz
    """
    with open(dat_path + '.pkl', 'wb') as output:
        pickle.dump(fd, output, -1)
