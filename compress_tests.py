import pandas as pd
from format_tools import load_object, save_object
from time import time
import numpy as np
import scipy as sp
from scipy import sparse

def convert_by_chunks(df, chunksize=500, dty=np.int16):
    blocks = []
    # Drop one block of columns at a time from the dataframe
    for i in range(df.shape[1]//chunksize):
        chunk = df.values[:, :chunksize]
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
    new_f = "test.npz"

    vals = np.random.rand(5000,5005)
    vals[vals < 0.77] = 0.
    df = pd.DataFrame(vals)
    save_object(df.index, "index_test.pkl")
    save_object(df.columns, "columns_test.pkl")
    df.reset_index(drop=True, inplace=True)
    df.columns = range(df.shape[1])  # takes up less space
    print("Dense memory usage:", df.memory_usage().sum()/1024**2, "MB")

    # Convert to sparse
    start = time()
    vals = convert_by_chunks(df)
    print(df.shape)
    del df
    inter1 = time()
    print("Converted to sparse in {} s".format(inter1 - start))
    size = vals.data.nbytes + vals.indices.nbytes + vals.indptr.nbytes
    print('{:.2f} MB of data'.format(size / 1024 ** 2))

    # Save as a hdf file for faster import later on
    #df.to_hdf(new_f, key='df', mode='w', compression=9, complib='blosc')
    sp.sparse.save_npz(new_f, vals)
    final = time()
    print("Finished saving as npz in {} s".format(final - inter1))
