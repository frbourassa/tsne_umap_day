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
# From an ndarray where different axes represent different conditions
# and observables are lined up on one axis.
def df_from_ndarray(ndarray, labels_dict_axis, observables_axis=-1,
                    obs_names=None, names=None):
    """
    Create a DataFrame from a multidimensional array in which all the data is.
    Each axis of the array, except for one, should correspond to a different
    experimental parameter (e.g. temperature, pressure, concentration,...),
    and each index along that axis corresponds to a different value of that
    parameter (e.g. 10 C, 20 C, 30 C, ...). The remaining axis corresponds to
    observables; each index along that axis is a different observable.
    Thus, specifying an index for each parameter axis fully specifies the
    experimental conditions and gives a vector of observables for that set
    of conditions.
    If you have multiple repeats for the same condition, one parameter axis
    can correspond to 'Repeats' or 'Samples'.
    This way of specifying the conditions assumes that experiments were made
    for all tuples in the cartesian product of the sets of parameter values.
    If some combinations were not tested, NaNs can be put in those slices.

    WARNING: will modify inplace labels_dict_axis and names.
    Input a copy if needed

    Args:
        ndarray (np.ndarray): the n-dimensional array containing all datapoints
        labels_dict_axis (dict of lists): dictionary of the parameter values
            along each axis that specifies an experimental parameter.
            Keys are the axes' numbers (axis 0, axis 1...).
            The list at key i should have the length of axis i of the array.
        observables_axis (int): the axis along which observables are specified.
        obs_names (list of str): the names corresponding to the
            observables along observables_axis.
        names (dict of str): the names of the parameter axes
            (e.g. 'temperature', 'pressure', etc. ).

    Returns:
        pd.DataFrame: the 2d DataFrame made by unraveling the axes
            corresponding to parameters/experimental conditions.
    """
    ob = observables_axis  # shorthand notation

    # Some dimensionality checks
    if ndarray.ndim - 1 < len(labels_dict_axis):
        raise ValueError(
            "There are not enough axes in ndarray for the inputted labels")
    if (obs_names is not None and ndarray.shape[ob] != len(obs_names)):
        raise ValueError( "There must be one observable name per element " +
            "on the observables_axis")
    for i in labels_dict_axis.keys():
        if i == (ob % ndarray.ndim):
            raise ValueError(
                "The observables should not be specified in labels_dict_axis")
        if ndarray.shape[i] != len(labels_dict_axis[i]):
            raise ValueError("The list of parameter values for axis {0} must\
                \n have the same length as this axis, which is {1}".format(
                    i, ndarray.shape[i]
                ))
    if (names is not None and len(names) != len(labels_dict_axis)):
        raise ValueError(
            "There must be one parameter name per axis in labels_dict_axis")
    if observables_axis > ndarray.ndim - 1:
        raise ValueError("Specified observables_axis isn't an existing axis")

    nb_obs = ndarray.shape[ob]
    # Hopefully, dimensions are correct now. Assemble the DataFrame.

    # First, move the observables axis to the end, then unravel other axes
    # so each row is a sample. The tricky part will then be to form tuples
    # with the inputted conditions to identify each row. We need to understand
    # very well how reshape works when dimensions are reduced.

    # Cyclic permutation to the right: the number of increments
    # is given by the following formula:
    shift = ((ndarray.ndim - 1) - ob) % ndarray.ndim
    # With numpy, one increment to the right at a time
    for i in range(shift):
        ndarray = np.moveaxis(ndarray, -1, 0)

    # We also need to permute the indices in dictionary keys
    labels_dict_axis = {((i + shift) % ndarray.ndim):labels_dict_axis[i]
                        for i in labels_dict_axis.keys()}
    if names is None:
        names = {}
        for i in range(ndarray.ndim - 1):
            original_i = (i - shift) % ndarray.ndim
            names[i] = "Axis {}".format(original_i)
    else:
        names = {((i + shift) % ndarray.ndim):names[i]
                for i in names.keys()}

    # Check if some axes are left unnamed. Create labels for them then.
    if len(labels_dict_axis) < ndarray.ndim - 1:
        for i in range(ndarray.ndim - 1):  # exclude last axis (observables)
            if i not in labels_dict_axis.keys():
                labels_dict_axis[i] = range(ndarray.shape[i])
            if i not in names.keys():
                # Happens only if names was not None initially
                # Original axis number, for identification
                original_i = (i - shift) % ndarray.ndim
                names[i] = "Axis {}".format(original_i)

    # When reshaping, inner levels are kept together first. If we have 3 axes,
    # and we reshape to two axes, then adjacent rows along axis 1 will be kept
    # together, then different arrays along axis 0 (outer) will be stacked.
    # So labels for inner axes vary faster (at each row, for axis -2).
    number_samples = np.prod(ndarray.shape[:-1])
    ndarray = ndarray.reshape(number_samples, nb_obs)

    # use from_product. The sortorder argument can be left to default
    # because we have reindexed the label dictionaries. We need lists
    labels_list_axis = [labels_dict_axis[i] for i in sorted(labels_dict_axis)]
    names_list = [names[i] for i in sorted(names)]
    idx = pd.MultiIndex.from_product(labels_list_axis, names=names_list)

    if obs_names is not None:
        cols = pd.Index(obs_names, name="Observables")
    else:
        cols = pd.Index(range(nb_obs), name="Observables")

    # Use the MultiIndex to index the 2d ndarray
    return pd.DataFrame(ndarray, index=idx, columns=cols)

# From multiple 2darrays corresponding to groups of sample points
def df_from_blocks(arrays, labels, observables=None, names=[]):
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
            by labels = ["10 C", "20 C", ...]). If labels are tuples, then
            names should have the same length as the tuples
            (to have one name per level).
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
    if (observables is not None and nb_observables != len(observables)):
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
    # Construct indexes for the samples in each array
    frames = []
    for i in range(len(arrays)):
        idx = pd.Index(range(arrays[i].shape[0]), name="Sample")
        frames.append(pd.DataFrame(arrays[i], columns=observables, index=idx))
    df = pd.concat(frames, keys=labels, names=names)

    # If there is only one sample per condition,
    # we don't want the "Sample" level since it is redundant
    if len(df.index.levels[-1]) == 1:
        df.reset_index(level=-1, inplace=True, drop=True)

    return df

# To add more information in the DataFrame index by regrouping labels
# of some level under another level.
def regroup_levels(frame, groups, level_group=None, axis=1, name=None):
    """
    In other words, add a level to columns or index of the dataframe
    by grouping the labels of a preexisting level.

    Args:
        frame (pd.DataFrame): the DataFrame to index better.
        groups (dict): a dictionary where keys are labels for the new grouping
            level, and values are lists of labels of the level to group that
            should be grouped under their key.
        level_group (str or int): the level to group
        axis (int): 0 or 1, rows or columns. 1 by default.
        name (str): the name of the new grouping level. Optional
    """
    # Check if the axis to group is Index (only one level) or MultiIndex
    idx = frame.index if axis == 0 else frame.columns
    if isinstance(idx, pd.MultiIndex):
        if level_group is None:
            raise ValueError(
                "The index on this axis is a MultiIndex; must specify a level")
        # The order of levels will be altered if an inner level is grouped
        original_names = list(idx.names)
        try:
            original_names.remove(level_group)
        except ValueError:
            raise ValueError("{} is not a valid level in the \
                DataFrame".format(level_group))
    # If we only have an Index, not a MultiIndex
    else:
        level_group = None
        original_names = []  # We have an Index

    # Split the dataframe, then concatenate with a new level
    blocks = {}
    for gp in groups.keys():
        subgroups = {k:frame.xs(k, level=level_group, axis=axis)
                    for k in groups[gp]}
        blocks[gp] = pd.concat(subgroups, names=[level_group], axis=axis, copy=False)

    # Update the order of the level names, return the concatenation of blocks
    final_names = [name, level_group] + original_names
    return pd.concat(blocks, axis=axis, names=final_names, copy=False)
