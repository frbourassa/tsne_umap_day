import pandas as pd
from format_tools import load_object, save_object

if __name__ == "__main__":
    fname = "data/GSE102827/GSE102827_frame_formatted_dense.pkl"
    new_f = "data/GSE102827/GSE102827_frame_formatted_sparse.h5"
    df = load_object(fname)
    df = df.to_sparse(fill_value=0)

    # Save as a hdf file for faster import later on
    df.to_hdf(new_f, key='df', mode='w')
