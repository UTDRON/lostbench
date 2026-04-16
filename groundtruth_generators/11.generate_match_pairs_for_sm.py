'''
Persist Match pairs in csv file for each test case pairs
'''
import pandas as pd
import os
import json

def main():
    
    input_df = pd.read_csv(GT_FILE)
    output_df = pd.DataFrame(columns = ["schema1", "schema2", "num_matches", "match_pairs"])

    with open(LINKS_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        links_lookup = json.load(f)

    for i in range(len(input_df)):
        matches = []
        schema1 = input_df["query_table"][i]
        df_schema1 = pd.read_csv(os.path.join(DATALAKE_PATH, schema1))
        columns1 = list(df_schema1.columns)

        schema2 = input_df["data_lake_table"][i]
        df_schema2 = pd.read_csv(os.path.join(DATALAKE_PATH, schema2))
        columns2 = list(df_schema2.columns)

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
        
        output_df.loc[len(output_df)] = {
            "schema1": schema1,
            "schema2": schema2,
            "num_matches": len(matches),
            "match_pairs": matches
        }
        if i % 500 == 0:
            print(f"{i+1} DONE! out of {len(input_df)}")
    
    return output_df

def check_one_to_many(lst):
    a_lst = []
    b_lst = []

    for a, b in lst:
        a_lst.append(a)
        b_lst.append(b)

    has_repeat = len(a_lst) != len(set(a_lst)) or len(b_lst) != len(set(b_lst))
    return has_repeat

if __name__ == "__main__":
    opened_schemas = {}
    GT_FILE = "../groundtruth/groundtruth_sm.csv"
    LINKS_LOOKUP_FILE = "../groundtruth/links_lookup.json"
    DATALAKE_PATH = "../LoStBench/datalake/"
    GT_ATTR_PAIRS = "../groundtruth/groundtruth_sm_attribute_pairs.csv"

    output_df = main()

    #removal of one to many match cases
    mask = output_df.apply(lambda r: check_one_to_many(r["match_pairs"]), axis=1)
    output_df = output_df[~mask]

    output_df["match_pairs"] = output_df["match_pairs"].apply(json.dumps)
    output_df.to_csv(GT_ATTR_PAIRS, index=False)
    print(f"Saved successfully to {GT_ATTR_PAIRS}")
