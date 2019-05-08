import pandas as pd
from format_tools import load_object, save_object
from time import time

if __name__ == "__main__":
    fname = "data/GSE102827/GSE102827_frame_formatted_dense.pkl"
    new_f = "data/GSE102827/GSE102827_frame_formatted_sparse.h5"

    # Load the object
    start = time()
    df = load_object(fname)
    inter1 = time()
    df = df.loc[:1000, :1000]  # Make the testing faster.
    print("Loaded the object in {} s".format(inter1 - start))
    print(df.shape)
    print("Dense memory usage:", df.memory_usage.sum()/1024**2, "MB")
    # Convert to sparse
    df = df.to_sparse(fill_value=0)
    inter2 = time()
    print("Converted to sparse in {} s".format(inter2 - inter1))
    print("Sparse memory usage:", df.memory_usage.sum()/1024**2, "MB")


    # Save as a hdf file for faster import later on
    df.to_hdf(new_f, key='df', mode='w', compression=9, complib='blosc')
    final = time()
    print("Finished saving as HDF5 in {} s".format(final - inter2))
