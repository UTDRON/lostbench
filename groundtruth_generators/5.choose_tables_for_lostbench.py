'''
For choosing tables for LoStBench (out of 2056 total)
1) Keeping only those tables that have >= 2 columns and <= 50 columns: (125 Removed) Remains 1931
2) Removing those tables that have columns that are 
completely empty- no values in the entire column: Further 535 to remove
Remaining: 2056-125-535 = 1396 remain
3) Further 10 removed manually(santos FD issue) remain: 1386
3) Finally Filtering those pairs from groundtruth_true_only.csv which are not in lostbench

'''
import pandas as pd
import os
import shutil

def main():
    cnt_copied = 0
    src_tables = os.listdir(SOURCE_TABLES_PATH)
    src_tables = [table for table in src_tables if table != '.DS_Store']

    for table in src_tables:
        n_cols = len(pd.read_csv(
            os.path.join(SOURCE_TABLES_PATH, table), 
            nrows=0
        ).columns) 
        if n_cols >= 2 and n_cols <= 50:
            df = pd.read_csv(os.path.join(SOURCE_TABLES_PATH, table), na_values=["", " ", "NA", "null", "NULL", "None","none"])
            empty_columns = df.columns[df.isna().all()].tolist() # finding columns that are completely empty
            if not empty_columns:
                shutil.copy2(os.path.join(SOURCE_TABLES_PATH, table), os.path.join(LOSTBENCH_PATH, table))
                cnt_copied += 1
        
    print(f"Copied {cnt_copied} files to Lostbench directory Successfully")

def filter_non_lostbench_tables():
    lostbench_tables = os.listdir(LOSTBENCH_PATH)
    lostbench_tables = [table for table in lostbench_tables if table != '.DS_Store']

    df = pd.read_csv(GT_TRUE_PATH)
    # Keep only rows where BOTH columns are in given_lst
    df_filtered = df[
        df["query_table"].isin(lostbench_tables) & df["data_lake_table"].isin(lostbench_tables)
    ].reset_index(drop=True)
    df_filtered.to_csv(GT_TUS_PATH, index=False)
    print(f"Saved filtered true table pairs groundtruth to {GT_TUS_PATH} successfully")


if __name__ == "__main__":
    '''
    Change these paths based on your codebase directory
    '''
    SOURCE_TABLES_PATH = f"../../../release_data/data_instances/type1_null_filled/tables/"
    LOSTBENCH_PATH = "../LoStBench/datalake/"
    GT_TRUE_PATH = "../groundtruth/groundtruth_true_only.csv"
    GT_TUS_PATH = "../groundtruth/groundtruth_tus_true_only.csv"

    # main()
    filter_non_lostbench_tables()

