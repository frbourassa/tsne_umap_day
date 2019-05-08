import pandas as pd
from format_tools import load_object, save_object
from time import time
import scipy as sp
from scipy import sparse

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
    idx, cols = df.index, df.columns
    vals = df.values
    dty = vals.dtype
    del df  # need to free memory, can't copy everything
    vals = sp.sparse.csc_matrix(vals, dtype=dty)
    inter2 = time()
    print("Converted to sparse in {} s".format(inter2 - inter1))
    size = vals.data.nbytes + vals.indices.nbytes + vals.indptr.nbytes
    print('{:.2f} MB of data'.format(size / 1024 ** 2))

    # Save as a hdf file for faster import later on
    sp.sparse.save_npz(new_f, vals)
    save_object(idx, "data/GSE102827/GSE102827_frame_index.pkl")
    save_object(cols, "data/GSE102827/GSE102827_frame_columns.pkl")
    final = time()
    print("Finished saving as npz in {} s".format(final - inter2))
