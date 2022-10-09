#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define paths to the data and results.

@author: Martin Goelz
"""
import os


def get_path_to_dat(fd_scen):
    """
    Return path to where data is stored.

    Parameters
    ----------
    fd_scen : str
        The scenario name.

    Returns
    -------
    str
        The path to the data.

    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(
        current_path, '..', 'data', fd_scen))


def get_path_to_res(fd_scen):
    """
    Return path to where results are stored.

    Parameters
    ----------
    fd_scen : str
        The scenario name.

    Returns
    -------
    str
        The path to the data.

    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(
        current_path, '..', 'results', fd_scen))
