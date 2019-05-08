import pandas as pd
from format_tools import load_object, save_object

if __name__ == "__main__":
    fname = "data/GSE102827/GSE102827_frame_formatted_dense.pkl"
    new_f = "data/GSE102827/GSE102827_frame_formatted_sparse.pkl"
    df = load_object(fname)
    df.to_sparse()
    save_object(df, new_f)
