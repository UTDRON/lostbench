'''
For the Generation of no of correspondences between columns
in each of the combination of pairs of tables in the dataset
table1, table2, correspondence?(y/n), num_correspondences
fields in csv: "query_table", "data_lake_table", "unionable", "num_union_cols"

outputs(csvs):
1. groundtruth_all
2. groundtruth with true only values for correspondence?(y/n) (or "unionable")

IMPORTANT: After this run "5.remove_reverse_duplicate_pairs.py" to avoid duplicate cases
'''
import pandas as pd
import os
import json

def main():
    with open(TABLE_COLUMNS_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        columns_lookup = json.load(f)

    with open(LINKS_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        links_lookup = json.load(f)

    tables = os.listdir(TABLES_PATH)
    tables.remove('.DS_Store')

    num_done = 0
    rows = []
    groundtruth = pd.DataFrame(columns=[
        "query_table", 
        "data_lake_table", 
        "unionable", 
        "num_union_cols"
        ]
    )

    for table1 in tables:
        for table2 in tables:
            if table1 != table2 and table1[5:9] != table2[5:9]:
                matches = []
                
                columns1 = columns_lookup[table1]
                columns2 = columns_lookup[table2]

                for col1 in columns1:
                    if col1 in links_lookup:
                        cumulative1 = links_lookup[col1]
                        cumulative1 = set(cumulative1)

                        for col2 in columns2:
                            if col2 in links_lookup:
                                cumulative2 = links_lookup[col2]
                                cumulative2 = set(cumulative2)

                                if cumulative1 & cumulative2:
                                    matches.append((col1,col2))
                            
                            elif col1 in columns2:
                                matches.append((col1,col2))

                    elif col1 in columns2:
                        matches.append((col1,col2))
                
                rows.append([table1, table2, 1 if matches else 0, len(matches)])
        num_done += 1
        print(f"{num_done} DONE! out of {len(tables)}")

    groundtruth = pd.DataFrame(rows, columns=[
        "query_table", "data_lake_table", "unionable", "num_union_cols"
    ])
    
    return groundtruth


if __name__ == "__main__":

    '''
    Change these paths based on your codebase directory
    '''
    TABLES_PATH = f"../../../release_data/data_instances/type1_null_filled/tables/"
    LINKS_LOOKUP_FILE = "../groundtruth/links_lookup.json"
    TABLE_COLUMNS_LOOKUP_FILE = "../groundtruth/table_columns_lookup.json"

    GT_ALL_OUTPUT_FILE = "../groundtruth/groundtruth_all.csv"
    GT_ONLY_TRUE_OUTPUT_FILE = "../groundtruth/groundtruth_true_only.csv"

    groundtruth = main()
    groundtruth.to_csv(GT_ALL_OUTPUT_FILE, index=False)
    print(f"Saved all linked table pairs groundtruth to {GT_ALL_OUTPUT_FILE} successfully")

    groundtruth_filtered = groundtruth[groundtruth["unionable"] == 1]
    groundtruth_filtered.to_csv(GT_ONLY_TRUE_OUTPUT_FILE, index=False)
    print(f"Saved True only cases  to {GT_ONLY_TRUE_OUTPUT_FILE} successfully")

    





