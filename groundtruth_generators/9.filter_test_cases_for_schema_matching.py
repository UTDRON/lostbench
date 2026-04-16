'''
For Chosing Test Case files for schema matching
1. no. of matches at least 50% of the smaller table
2. at most 50 columns per schema chosen


'''
import pandas as pd
import os

def get_col_count(path):
    if path not in col_count_cache:
        col_count_cache[path] = pd.read_csv(path, nrows=0).shape[1]
    return col_count_cache[path]


def keep_row(row):
    try:
        q_cols = get_col_count(os.path.join(DATALKE_PATH, row["query_table"]))
        d_cols = get_col_count(os.path.join(DATALKE_PATH, row["data_lake_table"]))

        return (
            row["num_union_cols"] >= 0.5 * min(q_cols, d_cols) #at least 50% of the smaller table
            and row["num_union_cols"] <= 50 #exclude the long unbreakable csvs in the dataset with hundreds of columns
            and q_cols <= 50  #no tables larger than 50 columns
            and d_cols <= 50
        )
    
    except Exception as e:
        print(e)
        return False
    

if __name__ == "__main__":
    col_count_cache = {}
    LABEL_FILE = "../groundtruth/groundtruth_tus_true_only.csv"
    GT_SMATCH_FILE = "../groundtruth/groundtruth_sm.csv"
    DATALKE_PATH = "../LoStBench/datalake/"

    df = pd.read_csv(LABEL_FILE)
    # filtered_df = df[df["query_table"].str.startswith("ANES_20", na=False)]
    filtered_df = df[df.apply(keep_row, axis=1)]
    
    print(len(df), "→", len(filtered_df))
    filtered_df.to_csv(GT_SMATCH_FILE, index = False)
