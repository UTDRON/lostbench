'''
Sampling for creating test case pairs for schema matching
1. at least 2 matches in schema pairs
2. sample with weighted probabilities 
3. a source schema should not appear more than once in test case pairs
4. Downsample to 450 

'''
import pandas as pd
import numpy as np

if __name__ == "__main__":

    GT_SM_FILE = "../groundtruth/groundtruth_sm_attribute_pairs.csv"
    GT_SM_TEST_FILE = "../groundtruth/groundtruth_sm_test_case.csv"

    df = pd.read_csv(GT_SM_FILE)
    df_filtered = df[df["num_matches"] >= 2].copy()

    # less weight for 2, higher weight for >2
    df_filtered["weight"] = df_filtered["num_matches"].apply(lambda x: 0.05 if x == 2 else 20.0)

    # sample per schema1, up to 2 rows with weighted probabilities
    def sample_group(g):
        n = min(len(g), 2)
        return g.sample(n=n, weights=g["weight"], random_state=42)

    df_limited = df_filtered.groupby("schema1", group_keys=False).apply(sample_group).reset_index(drop=True)

    # downsample to 600 rows
    if len(df_limited) > 600:
        final_df = df_limited.sample(n=600, random_state=42)
    else:
        final_df = df_limited
    
    # select 100 random rows with num_matches == 2 to drop them 
    to_drop = final_df[final_df["num_matches"] == 2].sample(n=150, random_state=42).index
    final_df = final_df.drop(index=to_drop).reset_index(drop=True)

    final_df.drop(columns=["weight"]).to_csv(GT_SM_TEST_FILE, index=False)

    print(final_df.value_counts("num_matches").sort_index())

    print(f"saved {len(final_df)} rows to {GT_SM_TEST_FILE} successfully!")
