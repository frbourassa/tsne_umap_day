import pandas as pd
from format_tools import load_object, save_object
from time import time
import numpy as np
import scipy as sp
from scipy import sparse

if __name__ == "__main__":
    new_f = "test.npz"

    vals = np.random.rand(5000,5000)
    vals[vals < 0.77] = 0.
    df = pd.DataFrame(vals)
    print("Dense memory usage:", df.memory_usage().sum()/1024**2, "MB")

    # Convert to sparse
    start = time()
    vals = sp.sparse.csc_matrix(df.values, dtype=df.values.dtype)
    inter1 = time()
    print("Converted to sparse in {} s".format(inter1 - start))
    print("Sparse memory usage:", df.memory_usage().sum()/1024**2, "MB")


    # Save as a hdf file for faster import later on
    #df.to_hdf(new_f, key='df', mode='w', compression=9, complib='blosc')
    sp.sparse.save_npz(new_f, vals)
    save_object(df.index, "index_test.pkl")
    save_object(df.columns, "columns_test.pkl")
    final = time()
    print("Finished saving as HDF5 in {} s".format(final - inter1))
