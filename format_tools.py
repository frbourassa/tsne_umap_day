"""
Module containing some wrapper functions to help organize your dataset into
a single DataFrame where each column contains an observable and each row,
a different sample, i.e. a vector in the phase space of observables.
@author: frbourassa
April 28, 2019
"""
import numpy as np
import pandas as pd
import pickle
import os

###
# Short functions to deal with saving/loading pickle files
###
def save_object(obj, filename):
    """ To save a Python object as a binary file using pickle.

    Warning: will overwrite any existing file under the same file name.

    Args:
        obj (object): the Python object to save in a pickle file
        filename (str): the file path and name (either a full path
            or a path relative to the current working directory).

    NB: it is common practice to use the extension .pkl for the file name.
    """
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """ Unpickle a file of pickled data.
    Args:
        filename (str): the file path and name (either a full path
            or a path relative to the current working directory)
    Returns:
        (object): the Python object saved in the filename.
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        return obj

###
# Functions to create a proper DataFrame
###
# From an ndarray where different axes represent different conditions or observables
def df_from_ndarray():

# From multiple 2darrays corresponding to blocks of sample points
def df_from_blocks(arrays, labels=[], observables=[], names=[]):
    """
    Create a DataFrame by stacking the 2darrays in the list. Columns of each
    array must correspond to the same observables, identified in the
    column_labels argument. Each 2darray correspond to a different subset of
    the dataset, whose conditions/properties are specified in the labels
    argument. The name argument names the property(ies) defining the blocks.

    Args:
        arrays (list of 2darrays): the arrays containing subsets of the data.
            They must have the same number of columns, since each column gives
            the value of one observable for each sample point.
        labels (list of str or int or tuples): list of the label(s) identifying
            the conditions corresponding to each block. Can be str/int if a
            single property identifies a block (e.g. temperature), or a tuple
            if more properties are needed (e.g. temperature and pressure). In
            that case, the DataFrame will have a MultiIndex.
        observables (list of str or int): optional. the name of the observable
            in each column (e.g. ["v_x", "v_y", "v_z", "r_x", ...]).
        names (list of str): optional. The name of the property(ies) that
            identify the blocks (e.g. ["Temperature" if blocks are specified
            by labels = ["10 C", "20 C", ...]).
    Returns:
        (pd.DataFrame): the DataFrame made of a vstack of arrays.
    """
    # Some dimensionality checks
    nb_observables = np.array([a.shape[1] for a in arrays], dtype=int)
    if not np.all(nb_observables == nb_observables[0]):
        raise ValueError(
            "All 2darrays must have the same number of elements along axis 1")
    else:
        nb_observables = nb_observables[0]  # now an int
    if (observables != [] and nb_observables[0] != len(observables)):
        raise ValueError("There must be one observable name per array column")

    nb_blocks = len(arrays)
    if nb_blocks != len(labels):
        raise ValueError(
            "There must be one label per 2darray in the arrays list")
    if type(labels[0]) in (list, tuple):
        nb_levels = np.array([len(a) for a in labels])
        if not np.all(nb_levels == nb_levels[0]):
            raise ValueError(
                "There must be the same number of properties in each label")
        if (names != [] and len(names) != len(labels[0])):
            raise ValueError(
                "There must be one name per property in each label tuple")

    # The inputted values should be OK now, hopefully.
    
