'''
The script in 4.linked_tables_gt_gen.py generates:
-> for every row_x: (a.csv,b.csv) there is a row_y: (b.csv,a.csv) 
-> these are reverse duplicate pairs because for both of these rows 
the values in the rest of the columns is same

-> To remove such reverse duplicate pairs run this script
'''
import pandas as pd

if __name__ == "__main__":
    '''
    Change these paths based on your codebase directory
    '''
    GT_SM = "../groundtruth/groundtruth_sm.csv"

    df = pd.read_csv(GT_SM)
    # Keep only rows where query_table < data_lake_table (lexicographically)
    df_dedup = df[df["query_table"] < df["data_lake_table"]].reset_index(drop=True)
    df_dedup.to_csv(GT_SM, index=False)
    print(f"Saved de-duplicated pairs to {GT_SM} successfully")


    '''
    Test code to count the reverse duplicate pairs
    '''
    # pairs = df_dedup.apply(
    #     lambda row: tuple(sorted([row["query_table"], row["data_lake_table"]])),
    #     axis=1
    # )
    # pair_counts = pairs.value_counts()
    # reverse_pair_count = pair_counts[pair_counts > 1].sum() // 2

    # print(reverse_pair_count)

    