# -*- coding: utf-8 -*-

#    pathpy is an OpenSource python package for the analysis of time series data
#    on networks using higher- and multi order graphical models.
#
#    Copyright (C) 2016-2018 Ingo Scholtes, ETH Zürich/Universität Zürich
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: scholtes@ifi.uzh.ch
#    Web:    http://www.ingoscholtes.net

import collections as _co
import random

import numpy as _np

from multiprocessing import Pool

from pathpy.utils import Log, Severity
from pathpy.classes.network import Network
from pathpy.classes.paths import Paths
from pathpy import algorithms

def random_walk(network, l, n=1, start_node=None):
    """
    [DEPRECATED]
    Generates n paths of a random walker in the given network
    and returns them as a paths object.
    Each path has a length of l steps.
    Parameters
    ----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The network structure on which the random walks will be simulated.
    int: l
        The (maximum) length of each random walk path. A path will
        terminate if a node with outdegree zero is reached.
    int: n
        The number of random walk paths to generate.
    """
    Log.add('The path_extraction.random_walk function is deprecated. Please use paths_from_random_walk instead.', Severity.WARNING)
    return paths_from_random_walk(network, l, n, start_node)


def paths_from_random_walk(network, l, n=1, start_node=None, expand_subpaths=True):
    """
    Generates n paths of a random walker in the given network
    and returns them as a paths object.
    Each path has a length of l steps.
    Parameters
    ----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The network structure on which the random walks will be simulated.
    int: l
        The (maximum) length of each random walk path. A path will
        terminate if a node with outdegree zero is reached.
    int: n
        The number of random walk paths to generate.
    """
    p = Paths()
    for i in range(n):
        path = algorithms.random_walk.generate_walk(network, l, start_node)
        p.add_path(tuple(path), expand_subpaths=expand_subpaths)
    return p

def random_paths(network, paths_orig, order=1, rand_frac=1.0, expand_subpaths=True, processes=1):
    """
    Generates Markovian paths of a random walker in a given network
    and returns them as a paths object.
    Parameters
    ----------
    network: Network
        The network structure on which the random walks will be simulated.
    paths_orig: Paths
        Paths that we want to randomise
    rand_frac: float
        The fraction of paths that will be randomised
    """
    p_rnd = Paths()
    params_list = []
    for l in paths_orig.paths:
        if l < order:
            continue

        for path, pcounts in paths_orig.paths[l].items():
            if pcounts[1] > 0:
                n_path = int(pcounts[1])
                n_path_rand = n_path
                if rand_frac < 1.0:
                    n_path_rand = _np.random.binomial(n_path, rand_frac)
                n_path_keep = n_path - n_path_rand

                if order == 1:
                    start_node = path[0]
                else:
                    start_node = network.separator.join(path[0:order])

                ## Add the random paths
                if n_path_rand > 0:
                    if processes == 1:
                        p_rnd += paths_from_random_walk(network, l, n_path_rand, start_node, expand_subpaths)
                    else:
                        params_list.append( [network, l, n_path_rand, start_node, expand_subpaths] )

                ## Keep the rest
                if n_path_keep > 0:
                    p_rnd.add_path(path, frequency=n_path_keep, expand_subpaths=expand_subpaths)

    if processes > 1:
        with Pool(processes) as p:
            results= p.starmap(paths_from_random_walk, params_list)
            for path in results:
                p_rnd += path

    return p_rnd
