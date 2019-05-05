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

# Trying to import from a pickle file first, then using the csv if not found
def import_singlecell(access_code, folder, raw_end, types_end):
    """ Function to call in all cases """
    # Check if the raw data was already imported and pickled
    file_raw_data = folder + access_code + raw_end
    raw_file_pickle = file_raw_data[:-4] + "_frame.pkl"
    try:
        df = load_object(raw_file_pickle)
    except FileNotFoundError:  # file not found, go below
        print("No pickle file was found; importing from csv")
        print("starting to count time")
        starting_time = measure_time()
        df, inter_time = load_raw_csv(file_raw_data, raw_file_pickle,
                starting_time)
    else:  # If the file was found, stop here.
        print("picke file for the raw dataframe was found and loaded")
        print("starting to count time")
        starting_time = measure_time()
        inter_time = starting_time
    finally:
        gc.collect()  # make sure we don't have a leak.

    # Now that we have the raw data in a DataFrame, MultiIndex it.
    print("\nNow starting to reindex the DataFrame, importing cell_types")
    df = reindex_save_plain_df(df, access_code, folder, types_end,
        inter_time, starting_time)
    return df

# Importing raw data from csv
def load_raw_csv(file_raw_data, raw_file_pickle, starting_time):
    """ Function called if the csv file was not already pickled """
    ## Import row and column names separately, because not ints...
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

    # Intermediate time
    inter_time = measure_time()
    print("Time taken to import headers: {} s".format(inter_time - starting_time))

    ## Import the raw data with pandas, as a sparse, because a lot of zeros.
    df = csv_to_sparse(file_raw_data, chunksize=5100, fill=0,
            dtype=np.int16, engine="c", header=None, skiprows=[0],
            usecols=cols_use, index_col=None)
    print("\nDataFrame: ")
    print(df)
    print("\nMemory usage of the full data frame:")
    print(df.memory_usage(deep=True).sum()/1024**2, "MB")

    # Intermediate time
    inter_time2 = measure_time()
    print("\nTime taken to import raw data: {} s".format(inter_time2 - inter_time))

    ## Index the frame with the separately imported labels.
    df.set_index(gene_names, inplace=True)
    df.index.name = "Gene"
    print("\nRows index: \n", df.index)
    df.rename(columns=cell_names, inplace=True)
    df.columns.name = "Cell"
    print("Columns:\n", df.columns)
    del gene_names, cell_names  # free some memory

    # Save a copy, in case the program crashes
    save_object(df, raw_file_pickle)
    inter_time3 = measure_time()
    print("Time taken to add the index: {} s".format(inter_time3 - inter_time2))

    return df, inter_time3

def reindex_save_plain_df(df, access_code, folder, types_end,
        inter_time, start_time):
    """ MultiIndexing the raw data imported into a DataFrame """
    # Import the cell types file
    file_cell_types = folder + access_code + types_end
    celltypes = pd.read_csv(file_cell_types, dtype="category", engine='c')

    # Time this read_csv
    inter_time3 = measure_time()
    print("Time taken to import cell types: {} s".format(inter_time3 - inter_time))

    # Create a MultiIndex object from this DataFrame. Maybe memory runs out
    # here because of the very complicated index codes.
    celltypes_index = pd.MultiIndex.from_frame(celltypes)
    del celltypes  # free some memory
    celltypes_index.rename("Cell", level=0, inplace=True)
    print("\nNew MultiIndex: ")
    print(celltypes_index)

    inter_time4 = measure_time()
    print("Time taken to create MultiIndex of cell types: {} s".format(inter_time4 - inter_time3))

    # Transpose the frame: genes are now columns
    # MultiIndex the cells with the cell type assignment.
    df = df.T
    inter_time5 = measure_time()
    print("Time taken to transpose: {} s".format(inter_time5 - inter_time4))

    # Set the index to the multiIndex.
    df.set_index(celltypes_index, inplace=True)
    inter_time6 = measure_time()
    print("Time taken to multiIndex rows: {} s".format(inter_time6 - inter_time5))

    # Reorder the index by stimulation time. May take a while to do this.
    df.sort_index(axis=0, level="stim", inplace=True)
    inter_time7 = measure_time()
    print("Time taken to sort rows: {} s".format(inter_time7 - inter_time6))

    # Put the cell number at the end
    for k in range(df.index.nlevels - 1):
        df = df.swaplevel(axis=0, i=k, j=k + 1)
    inter_time8 = measure_time()
    print("Time taken to shuffle cell names to last level:" +
        "{} s".format(inter_time8 - inter_time7))

    # Print the result of our good work
    print("\nFull formatted DataFrame:")
    print(df)
    print("\nMemory usage now:")
    print(df.memory_usage(deep=True).sum()/1024, "kB")

    # Save the full formatted dataFrame in a pickle file
    save_object(df, folder + access_code + "_frame_formatted.pkl")

    ## Also, save blocks of the frame, separated by stimulation time,
    # to mimic the situation where we want to stack blocks of dfs.
    different_times = list(df.index.get_level_values("stim").drop_duplicates())
    print("Stimulation time of each block")
    print(different_times)

    inter_time9 = measure_time()
    print("Time taken to save formatted df " +
        "and list stimulation times: {} s".format(inter_time9 - inter_time8))

    # Now, save each stimulation value block separately.
    blocks = [df.loc[a] for a in different_times]
    inter_time10 = measure_time()
    print("Time taken to split df in blocks: {} s".format(
                                                inter_time10 - inter_time9))

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
    elapsed_time = final_time - start_time
    print("The whole operation lasted {} s".format(elapsed_time))

    return df

if __name__ == "__main__":
    # Files to use
    access_code = "GSE102827"
    folder = 'data/' + access_code + "/"

    raw_end = '_merged_all_raw.csv'
    types_end = '_cell_type_assignments.csv'
    #types_end = "_cell_type_test.csv"
    #raw_end = "_raw_test.csv"

    import_singlecell(access_code, folder, raw_end, types_end)
