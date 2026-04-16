'''
TUS Benchmark generator for ANES from groundtruth
'''
import pandas as pd
import os
import pickle
import json


def main():

    gt_df = pd.read_csv(GT_FILE)
    query_tables = [f for f in os.listdir(QUERY_DIR) if f.endswith(".csv")]

    anesUnionBenchmark = {}
    for table in query_tables:
        matched = gt_df.loc[gt_df["query_table"] == table, "data_lake_table"].tolist()
        anesUnionBenchmark[table] = matched
        # anesUnionBenchmark[table].append(table)

    with open(JSON_OUT, "w") as f:
        json.dump(anesUnionBenchmark, f, indent=2, ensure_ascii=False)
    print(f"Saved dictionary to {JSON_OUT}")

    with open(PICKLE_OUT, "wb") as f:
        pickle.dump(anesUnionBenchmark, f)
    print(f"Saved dictionary to {PICKLE_OUT}")

    #adding self as candidate in groundtruth
    for table in query_tables:
        anesUnionBenchmark[table].append(table)

    with open(JSON_OUT2, "w") as f:
        json.dump(anesUnionBenchmark, f, indent=2, ensure_ascii=False)
    print(f"Saved dictionary to {JSON_OUT2}")

    with open(PICKLE_OUT2, "wb") as f:
        pickle.dump(anesUnionBenchmark, f)
    print(f"Saved dictionary to {PICKLE_OUT2}")


if __name__ == "__main__":
    GT_FILE = "../groundtruth/groundtruth_tus_true_only.csv"
    QUERY_DIR = "../LoStBench/query/"
    PICKLE_OUT = "../groundtruth/anesUnionBenchmark.pickle"
    PICKLE_OUT2 = "../groundtruth/anesUnionBenchmark_self_added.pickle"
    JSON_OUT = "../groundtruth/anesUnionBenchmark.json"
    JSON_OUT2 = "../groundtruth/anesUnionBenchmark_self_added.json"
    
    main()