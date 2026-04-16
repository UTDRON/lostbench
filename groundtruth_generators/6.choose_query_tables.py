'''
1. At first stratifically sampled ~25% tables from 1396 based on
no of unionable tables
2. Then lexicographically sorted the table_name[10:] part 
for getting similar table names as grouped together
3. Manually chose 120
'''
import pandas as pd
import shutil
import os


def main():
    '''
    1. Sample 25% stratifically from tables in datalake based on no of 
    unionable tables
    '''
    # df = pd.read_csv(GT_TUS_FILE)
    # counts = df['query_table'].value_counts()

    # sampled = counts.sample(
    #     n=350,
    #     weights=counts.values,
    #     random_state=42
    # )

    '''
    2. Lexicographically sort the tables based on table_name[10:] part 
    for getting similar table names as grouped together
    '''
    # sampled_sorted = sampled.sort_index(key=lambda x: x.str[10:])

    '''
    3. Chose 115 manually fron the list printed below
    '''
    # for k, v in sampled_sorted.items():
    #     print(f"{k}: {v}")

    '''
    4. Copy those 115 to query folder
    '''
    with open(HANDPICKED_TABLES, "r", encoding="utf-8") as f:
        query_tables = [line.strip() for line in f if line.strip()]  # remove empty lines

    for table in query_tables:
        SRC_PATH = os.path.join("../LoStBench/datalake/", table)
        DST_PATH = os.path.join("../LoStBench/query/", table)
        shutil.copy2(SRC_PATH, DST_PATH)
    print(f"Copied {len(query_tables)} files Successfully to {DST_PATH}")


if __name__ == "__main__":
    GT_TUS_FILE = "../groundtruth/groundtruth_tus_true_only.csv"
    HANDPICKED_TABLES = "../groundtruth/query.txt"

    main()

