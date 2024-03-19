import pandas as pd

def load_data(ratio_cols):
    df = pd.read_excel('full_data.xlsx', header=0)
    df = df[ratio_cols]
    return df