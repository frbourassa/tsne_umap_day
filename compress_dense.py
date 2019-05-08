import pandas as pd
from format_tools import load_object, save_object
from time import time
import scipy as sp
from scipy import sparse
import numpy as np


def convert_by_chunks(df, chunksize=500, dty=np.int16):
    blocks = []
    # Drop one block of columns at a time from the dataframe
    for i in range(df.shape[1]//chunksize):
        chunk = df.values[:, :chunksize]
        print("Saved a chunk")
        blocks.append(sp.sparse.csc_matrix(chunk, dtype=dty))
        df.drop(labels=range(i*chunksize, (i+1)*chunksize), axis=1, inplace=True)
        print("Chunk {} done".format(i))

    # remaining part: rename columns so we know hwere to start
    df.columns=range(df.shape[1])
    blocks.append(sp.sparse.csc_matrix(df.values, dtype=dty))
    df.drop(labels=range(df.shape[1]), inplace=True)

    # Put the blocks together
    return sp.sparse.hstack(blocks, dtype=dty)

if __name__ == "__main__":
    fname = "data/GSE102827/GSE102827_frame_formatted_sparse.pkl"
    new_f = "data/GSE102827/GSE102827_frame_values_sparse.npz"

    # Load the object
    start = time()
    df = load_object(fname)
    inter1 = time()
    print("Loaded the object in {} s".format(inter1 - start))
    print("Shape: ", df.shape)
    print("Type: ", type(df))
    print("DataFrame memory usage:", df.memory_usage().sum()/1024**2, "MB")

    # Save the values to a sparse matrix, save the indexes separately
    inter2 = time()
    save_object(df.index, "data/GSE102827/GSE102827_frame_index.pkl")
    save_object(df.columns, "data/GSE102827/GSE102827_frame_columns.pkl")
    df.reset_index(drop=True, inplace=True)
    df.columns = range(df.shape[1])  # takes up less space
    dty = np.int16
    print("Starting convert_by_chunks")
    vals = convert_by_chunks(df, chunksize=5000, dty=dty)
    del df

    print("Converted to sparse in {} s".format(inter2 - inter1))
    size = vals.data.nbytes + vals.indices.nbytes + vals.indptr.nbytes
    print('{:.2f} MB of data'.format(size / 1024 ** 2))

    # Save as a npz file for faster import later on
    sp.sparse.save_npz(new_f, vals)
    final = time()
    print("Finished saving as npz in {} s".format(final - inter2))
