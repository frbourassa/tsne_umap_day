import pandas as pd
import numpy as np
import scipy as sp
from format_tools import load_object, save_object, df_from_ndarray, df_from_blocks, regroup_levels

def import_2d_excel(folder, fname):
    try:
        df = pd.read_excel(folder + fname, index_col=0)
    except FileNotFoundError as e:
        print(e)
        print("File could  not be found; returning None")
        df = None

    return df

if __name__ == "__main__":
    folder = "data/ndarrays/"
    fname = "CoreDen-PathwayScores.xlsx"

    df = import_2d_excel(folder, fname).T
    df.index.name = "Patients"
    df.sort_index(inplace=True)
    print(df)

    # Another file available: same patients, other pathways
    fname = "EpiDen-PathwayScores.xlsx"

    df2 = import_2d_excel(folder, fname).T
    df2.index.name = "Patients"
    df2.sort_index(inplace=True)
    print(df2)


    #assert df2.index == df.index

    # Stack the two dataframes together: horizontal Stack
    # Add an extra level to regroup genes: Core or Epi

    df_tot = pd.concat([df, df2], axis=1, keys=["Core", "Epi"], names=["Importance"], sort=True)
    df_tot.index.name = "Patient"
    print(df_tot)
    print(df_tot.columns)
    print(df_tot.index)

    save_object(df, 'data/bulk_gene_data_core_felix.pkl')
    save_object(df2, 'data/bulk_gene_data_epi_felix.pkl')
    save_object(df_tot, 'data/bulk_gene_data_concatenated_felix.pkl')
