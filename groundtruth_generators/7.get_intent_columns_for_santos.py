'''
For getting intent column for SANTOS experiment
the column having highest no of correspondences a table is its intent column
'''
import pandas as pd
import os
import json
import pickle

def main():
    num_done = 1
    record = {}

    query_tables = os.listdir(QUERY_PATH)
    query_tables = [table for table in query_tables if table != '.DS_Store']

    with open(TABLE_COLUMNS_LOOKUP, "r", encoding = "utf-8") as f:
        tc_lookup = json.load(f)
    
    with open(LINKS_LOOKUP, "r", encoding = "utf-8") as f:
        links_lookup = json.load(f)

    gt_df = pd.read_csv(GT_FILE)
    
    for table1 in query_tables:
        print(f"{num_done}. Doing for: {table1}")

        columns1 = tc_lookup[table1]
        matched = gt_df.loc[gt_df["query_table"] == table1, "data_lake_table"].to_list()

        record[table1] = {}
        for col1 in columns1:
            record[table1][col1] = 0
        
        for table2 in matched:
            columns2 = tc_lookup[table2]

            for col1 in columns1:
                if col1 in links_lookup:
                    cumulative1 = links_lookup[col1]
                    cumulative1 = set(cumulative1)

                    for col2 in columns2:
                        if col2 in links_lookup:
                            cumulative2 = links_lookup[col2]
                            cumulative2 = set(cumulative2)

                            if cumulative1 & cumulative2:
                                record[table1][col1] += 1
                        
                        elif col1 in columns2:
                            record[table1][col1] += 1
                
                elif col1 in columns2:
                    # print(record[table1])
                    record[table1][col1] += 1

        num_done +=1
    
    return record

def get_highest_keys():
    
    record = main()
    highest_keys = {}
    for csv_name, cols in record.items():
        if cols: 
            max_key = max(cols, key=cols.get)
            highest_keys[csv_name] = max_key

    with open(HIGHEST_KEYS_OUTPUT, "w") as f:
        json.dump(highest_keys, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"Highest Keys Saved to {HIGHEST_KEYS_OUTPUT}")

    return highest_keys



def save_intent_col_benchmark():
    benchmark = {}
    for csv_name, col_name in highest_keys.items():

        try:
            df = pd.read_csv(os.path.join(QUERY_PATH, csv_name), nrows=0)
            if col_name in df.columns:
                idx = df.columns.get_loc(col_name)
                benchmark[csv_name[:-4]] = int(idx)
            else:
                print(f"Column '{col_name}' not found in {csv_name}")
        except Exception as e:
            print(e)
    
    with open(INTENT_COLUMN_JSON, "w") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
    print()
    print(f"Intent Column Benchmark Saved to {INTENT_COLUMN_JSON}")

    with open(INTENT_COLUMN_PICKLE, "wb") as f:
        pickle.dump(benchmark, f)
    print(f"Intent Column Benchmark Pickle Saved to {INTENT_COLUMN_PICKLE}")
        

if __name__ == "__main__":
    GT_FILE = "../groundtruth/groundtruth_tus_true_only.csv"
    QUERY_PATH = "../LoStBench/query/"
    DATALAKE_PATH = "../LoStBench/datalake/"
    TABLE_COLUMNS_LOOKUP = "../groundtruth/table_columns_lookup.json"
    LINKS_LOOKUP = "../groundtruth/links_lookup.json"
    HIGHEST_KEYS_OUTPUT = "../groundtruth/highest_keys.json"
    INTENT_COLUMN_JSON = "../groundtruth/anesIntentColumnBenchmark.json"
    INTENT_COLUMN_PICKLE = "../groundtruth/anesIntentColumnBenchmark.pickle"

    highest_keys = get_highest_keys()

    save_intent_col_benchmark()

    
