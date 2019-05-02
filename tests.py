import numpy as np
import pandas as pd

from format_tools import df_from_blocks, df_from_ndarray, regroup_levels

def test_ndimarray():
    # Setup a simple example: conditions are T and p, obs are first axis
    arr = np.arange(3*4*5).reshape(3, 4, 5)
    obs_axis = 0
    names = {1:"Temperature", 2:"Pressure"}
    param_labels = {
        1: ["10 C", "20 C", "30 C", "40 C"],
        2: ["{} atm".format(i) for i in range(5)]
    }
    obs_names = ['vx', 'vy', 'vz']

    # Try some faulty inputs first
    try:
        df_from_ndarray(arr[:, :, 0], param_labels, observables_axis=obs_axis,
                    obs_names=obs_names, names=names)
    except ValueError as e: print(e)
    try:
        df_from_ndarray(arr[:, :, 1:4], param_labels, observables_axis=obs_axis,
                    obs_names=obs_names, names=names)
    except ValueError as e: print(e)
    try:
        df_from_ndarray(arr, param_labels, observables_axis=1,
                    obs_names=obs_names, names=names)
    except ValueError as e: print(e)
    try:
        param_labels2 = {i:param_labels[i] for i in sorted(param_labels)[:-1]}
        df_from_ndarray(arr, param_labels2, observables_axis=obs_axis,
                    obs_names=obs_names, names=names)
    except ValueError as e: print(e)
    try:
        df_from_ndarray(arr, param_labels, observables_axis=obs_axis,
                    obs_names=obs_names[1:], names=names)
    except ValueError as e: print(e)
    try:
        names2 = {i:names[i] for i in sorted(names)[:-1]}
        df_from_ndarray(arr, param_labels, observables_axis=obs_axis,
                    obs_names=obs_names, names=names2)
    except ValueError as e: print(e)

    # Correct input: look at what it gives to know if it is as expected.
    # i.e. the first observable should have contiguous values in the first
    # column of the df, because all vx values are in the first element
    # of axis 0. Pressure should be an inner level, temperature an outer one.
    df = df_from_ndarray(arr, param_labels, observables_axis=obs_axis,
                obs_names=obs_names, names=names)
    print(df)

    # Default values should not bug,
    # and parameter axes not specified should be titled "Axis i"
    df = df_from_ndarray(arr, {})
    print(df)

    # Another try: observables are in the last axis, so contiguous.
    arr = np.arange(3*4*5).reshape(4, 5, 3).astype(float)
    arr[1, 2:4, :] = np.array([[np.nan]*3]*2)
    obs_axis = -2
    obs_names2 = ["vx", 'vy', 'vz', 'dum', 'my']
    names = {0:"Temperature", 1:"Pressure"}

    # A faulty input: the obs axis is an axis in the param_labels dict
    try:
        df_from_ndarray(arr, param_labels, observables_axis=obs_axis,
                    obs_names=obs_names2, names=names)
    except ValueError as e: print(e)

    param_labels = {
        0: ["10 C", "20 C", "30 C", "40 C"],
        1: ["{} atm".format(i) for i in range(5)]
    }
    # Now it should be fine
    obs_axis = -1
    df = df_from_ndarray(arr, param_labels, observables_axis=obs_axis,
                obs_names=obs_names, names=names)
    print(df)

    return 0

def test_blocks():
    # Test the input by block function
    blocks = [np.random.rand(5, 5) for i in range(3)]  # 3 blocks
    temperatures = [2, 4, 6]
    pressures = [101, 201, 301]
    labels = tuple([(temperatures[i], pressures[i]) for i in range(3)])
    observables = ["A", "B", "C", "D", "E"]

    # Test some wrong inputs
    names = ["Temperature"]
    try:
        df_from_blocks(blocks, labels=labels, observables=observables, names=names)
    except ValueError as e:
        print(e)

    names.append("Pressure")
    try:
        df_from_blocks(blocks, labels=labels, observables=observables[:3], names=names)
    except ValueError as e:
        print(e)
    try:
        df_from_blocks(blocks[:2], labels=labels, observables=observables, names=names)
    except ValueError as e:
        print(e)
    try:
        df_from_blocks(blocks, labels=labels, observables=observables, names=names[1:])
    except ValueError as e:
        print(e)

    # Try the correct input
    df = df_from_blocks(blocks, labels=labels, observables=observables, names=names)
    print(df)
    assert df.index.names == names + ['Sample'], "from_blocks creates the wrong index"

    # Try single-row arrays
    blocks = [np.random.rand(1, 5) for i in range(3)]  # 3 blocks
    df = df_from_blocks(blocks, labels=labels, observables=observables, names=names)
    assert len(df.index.names) == 2, "Deletion of useless level does not work"
    print(df)

    df_from_blocks(blocks, labels=labels)

def test_regroup():
    # Create a dataframe first.
    # Conditions are T and p, obs are first axis
    arr = np.arange(6*4*5).reshape(6, 4, 5)
    obs_axis = 0
    names = {1:"Temperature", 2:"Pressure"}
    param_labels = {
        1: ["10 C", "20 C", "30 C", "40 C"],
        2: ["{} atm".format(i) for i in range(5)]
    }
    obs_names = ['vx', 'vy', 'vz'] + ['x', 'y', 'z']
    df = df_from_ndarray(arr, param_labels, obs_axis, obs_names, names)

    # Regrouping x, y, and z coordinates
    groups = {"X":['x', 'vx'], "Y":['y', 'vy'], "Z":['z', 'vz']}
    ret = regroup_levels(df, groups, level_group="Observables", axis=1, name="Dimension")
    print(ret)
    print(ret.columns)

    # Regrouping temperatures by heat.
    groups = {"cold":['10 C', '20 C'], "hot":['30 C', '40 C']}
    ret = regroup_levels(df, groups, level_group="Temperature", axis=0, name="Feeling")
    print(ret)

    # Regrouping pressures by effect on a human
    groups = {"burst":['0 atm'], "fine":["1 atm"],
        "faint":["2 atm"], "crush":['3 atm', '4 atm']}
    #print(df.xs("0 atm", level="Pressure", axis=0))
    ret = regroup_levels(df, groups, level_group="Pressure", axis=0, name="Effect")
    print(ret)

if __name__ == "__main__":
    #test_blocks()
    #test_ndimarray()
    test_regroup()
