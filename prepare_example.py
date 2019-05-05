# -*- coding:utf-8 -*-
"""
@author: frbourassa
"""
import numpy as np
import scipy as sp
import pandas as pd
from format_tools import csv_to_sparse, load_object, save_object
from time import time as measure_time
import gc

if __name__ == "__main__":
    starting_time = measure_time()
    # Files to use
    access_code = "GSE102827"
    folder = 'data/' + access_code + "/"
    file_raw_data = folder + access_code + '_merged_all_raw.csv'
    file_cell_types = folder + access_code + '_cell_type_assignments.csv'

    # Import row and column names separately, because not ints...
    # Genes are rows, cells are columns (the opposite of what we expect)
    # Gene names are in the first column
    gene_names = pd.read_csv(file_raw_data, dtype=str, usecols=[0],
            engine='c', skiprows=0, squeeze=True, na_filter=False)
    cell_names = pd.read_csv(file_raw_data, nrows=1, dtype=str,
            engine='c', header=None, na_filter=False).iloc[0, 1:]
    print("Gene names: ")
    print(gene_names)
    print("\nCell names: ")
    print(cell_names)
    cols_use = range(1, cell_names.size + 1)

    # Import the raw data with pandas, as a sparse, because a lot of zeros.
    df = csv_to_sparse(file_raw_data, chunksize=5100, fill=0,
            dtype=np.int16, engine="c", header=None, skiprows=[0],
            usecols=cols_use, index_col=None)
    print("\nMemory usage of the full data frame:")
    print(df.memory_usage(deep=True).sum()/1024**2, "MB")
    print("\nDataFrame: ")
    print(df)

    # Index the frame with the separately imported labels.
    df.set_index(gene_names, inplace=True)
    df.index.name = "Gene"
    print(df.index)
    df.rename(columns=cell_names, inplace=True)
    df.columns.name = "Cell"
    print(df.columns)

    # Save a copy, in case the program crashes
    save_object(df, file_raw_data)
    gc.collect()  # make sure we don't have a leak. 
    print("Now starting to reindex the DataFrame, importing cell_types")

    # Import the attributed cell types, which we'll use as row index.
    celltypes = pd.read_csv(file_cell_types, dtype="category")
    celltypes_index = pd.MultiIndex.from_frame(celltypes)
    celltypes_index.rename("Cell", level=0, inplace=True)
    print("\nNew MultiIndex: ")
    print(celltypes_index)

    # Transpose the frame: genes are now columns
    # MultiIndex the cells with the cell type assignment.
    df = df.T
    df.set_index(celltypes_index, inplace=True)

    # Reorder the index by stimulation time
    df.sort_index(axis=0, level="stim", inplace=True)

    # Put the cell number at the end?
    for k in range(df.index.nlevels - 1):
        df = df.swaplevel(axis=0, i=k, j=k + 1)

    print("\nFull formatted DataFrame:")
    print(df)
    print("\nMemory usage now:")
    print(df.memory_usage(deep=True).sum()/1024, "kB")

    # Save the full formatted dataFrame in a pickle file
    #save_object(df, "data/" + access_code + "formatted.pkl")

    # Also, save blocks of the frame, separated by stimulation time,
    # to mimic the situation where we want to stack blocks of dfs.
    different_times = list(df.index.get_level_values("stim").drop_duplicates())
    print("Stimulation time of each block")
    print(different_times)

    # Now, save each stimulation value block separately.
    blocks = [df.loc[a] for a in different_times]
    for i, b in enumerate(blocks):
        bname = "data/blocks/" + access_code + "_stim_" + different_times[i] + ".pkl"
        save_object(b, bname)

    # Save information about the data, to be able to concat the blocks
    times_name = "data/blocks/" + access_code + "_stim_times.pkl"
    save_object(different_times, times_name)
    save_object(df.columns,
            "data/blocks/" + access_code + "_gene_names.pkl")

    # To know how long it took
    final_time = measure_time()
    elapsed_time = final_time - starting_time
    print("The whole operation lasted {} s".format(elapsed_time))
