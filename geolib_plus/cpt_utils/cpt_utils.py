"""
Tools for CPT tool
"""
import os
import sys

# import packages
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np

from geolib_plus.cpt_base_model import AbstractCPT

resource_default_path = Path(__file__).parent.parent.parent


def n_iter(
    n: Union[Iterable, None],
    qt: Union[Iterable, None],
    friction_nb: Union[Iterable, None],
    sigma_eff: Union[Iterable, None],
    sigma_tot: Union[Iterable, None],
    Pa: Union[Iterable, None],
) -> Iterable:
    """
    Computation of stress exponent *n*

    Parameters
    ----------
    :param n: initial stress exponent
    :param qt: tip resistance
    :param friction_nb: friction number
    :param sigma_eff: effective stress
    :param sigma_tot: total stress
    :param Pa: atmospheric pressure
    :return: updated n - stress exponent
    """

    # convergence of n
    Cn = (Pa / np.array(sigma_eff)) ** n

    Q = ((np.array(qt) - np.array(sigma_tot)) / Pa) * Cn
    F = (np.array(friction_nb) / (np.array(qt) - np.array(sigma_tot))) * 100

    # Q and F cannot be negative. if negative, log10 will be infinite.
    # These values are limited by the contours of soil behaviour of Robertson
    Q[Q <= 1.0] = 1.0
    F[F <= 0.1] = 0.1
    Q[Q >= 1000.0] = 1000.0
    F[F >= 10.0] = 10.0

    IC = ((3.47 - np.log10(Q)) ** 2.0 + (np.log10(F) + 1.22) ** 2.0) ** 0.5

    n = 0.381 * IC + 0.05 * (sigma_eff / Pa) - 0.15
    n[n > 1.0] = 1.0
    return n


def resource_path(file_name: Union[Path, str]) -> Path:
    """
    Define the relative path to the file

    Used to account for the compiling location of the shapefile

    Parameters
    ----------
    :param file_name: File name
    :return: relative path to the file
    """
    assert (
        resource_default_path.is_dir()
    ), f"Default resource path was not found at {resource_default_path}"
    return resource_default_path / file_name


def ceil_value(data: Iterable, value: Union[int, float]) -> Iterable:
    """
    Replaces the data values from data, that are are smaller of equal to value.
    It replaces the data values with the first non-zero value of the dataset.

    :param data:
    :param value:
    :return: data with the updated values
    """
    # collect indexes smaller than value
    idx = [i for i, val in enumerate(data) if val <= value]

    # get consecutive indexes on the list
    indx_conseq = []
    for k, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1]):
        indx_conseq.append(list(map(itemgetter(1), g)))

    # assigns the value of the first non-value
    for i in indx_conseq:
        for j in i:
            # if the sequence contains the last index of the data use the previous one
            if i[-1] + 1 >= len(data):
                data[j] = data[i[0] - 1]
            else:
                data[j] = data[i[-1] + 1]

    return data


def merge_thickness(cpt_data: AbstractCPT, min_layer_thick: int):
    """
    Reorganises the lithology based on the minimum layer thickness.
    This function call the functions merging_label, merging_index , merging_depth , merging_thickness.
    These functions merge the layers according to the min_layer_thick.
    For more information refer to those.

    Parameters
    ----------
    :param cpt_data : CPT data set
    :param min_layer_thick : Minimum layer thickness
    :return depth_json: depth merged
    :return indx_json: index of the merged list
    :return lithology_json: merged lithology

    """

    # variables
    lithology = cpt_data.lithology
    depth = cpt_data.depth

    # Find indices of local unmerged layers
    aux = ""
    idx = []
    for j, val in enumerate(lithology):
        if val != aux:
            aux = val
            idx.append(j)

    # Depth between local unmerged layers
    local_z_ini = [depth[i] for i in idx]
    # Thicknesses between local unmerged layers
    local_thick = np.append(np.diff(local_z_ini), depth[-1] - local_z_ini[-1])
    # Actual Merging
    new_thickness = merging_thickness(local_thick, min_layer_thick)

    depth_json = merging_depth(depth, new_thickness)
    indx_json = merging_index(depth, depth_json)
    lithology_json = merging_label(indx_json, lithology)

    return depth_json, indx_json, lithology_json


def merging_label(indx_json: Iterable, lithology: Iterable) -> List:
    """
    Function that joins the lithology labels of each merged layer.
    """
    new_label = []
    start = indx_json[:-1]
    finish = indx_json[1:]
    for i in range(len(start)):
        # sorted label list
        label_list = sorted(
            set(lithology[start[i] : finish[i]]),
            key=lambda x: lithology[start[i] : finish[i]].index(x),
        )
        new_label.append(r"/".join(label_list))
    return new_label


def merging_index(depth: Iterable, depth_json: Iterable, tol: float = 1e-12) -> List:
    """
    Function that produces the indexes of the merged layers by finding which depths are referred.
    """
    new_index = []

    for i in range(len(depth_json)):
        new_index.append(
            int(np.where(np.abs(depth_json[i] - np.array(depth)) <= tol)[0][0])
        )

    return new_index


def merging_depth(depth: Iterable, new_thickness: Iterable) -> Iterable:
    """
    Function that calculates the top level depth of each layer by summing the thicknesses.
    """
    new_depth = np.append(depth[0], new_thickness)
    new_depth = np.cumsum(new_depth)
    return new_depth


def merging_thickness(local_thick: Iterable, min_layer_thick: int) -> List:
    """
    In this function the merging og the layers is achieved according to the min_layer thick.

    .._element:
    .. figure:: ./_static/Merge_Flowchart.png
        :width: 350px
        :align: center
        :figclass: align-center

    """

    new_thickness = []
    now_thickness = 0
    counter = 0
    while counter <= len(local_thick) - 1:
        while now_thickness < min_layer_thick:
            now_thickness += local_thick[counter]
            counter += 1
            if int(counter) == len(local_thick) and now_thickness < min_layer_thick:
                new_thickness[-1] += now_thickness
                return new_thickness
        new_thickness.append(now_thickness)
        now_thickness = 0
    return new_thickness


def smooth(
    sig: Iterable, window_len: int = 10, lim: Optional[Union[int, float]] = None
):
    """
    Smooth signal

    It uses a moving average with window to smooth the signal

    :param sig: original signal
    :param window_len: (optional) number of samples for the smoothing window: default 10
    :param lim: (optional) limit the minimum value of the array: default None: does not apply limit.
    :return: smoothed signal
    """

    # if window length bigger that the size of the signal: window is the same size as the signal
    if window_len > len(sig):
        window_len = len(sig)

    s = np.r_[
        2 * sig[0] - sig[window_len:1:-1], sig, 2 * sig[-1] - sig[-1:-window_len:-1]
    ]
    # constant window
    w = np.ones(window_len)
    # convolute signal
    y = np.convolve(w / w.sum(), s, mode="same")
    # limit the value is exits
    if lim is not None:
        y[y < lim] = lim
    return y[window_len - 1 : -window_len + 1]
