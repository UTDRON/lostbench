import pandas as pd
import json

GT_AL_FILE = "../groundtruth/groundtruth_true_only.csv"
GT_LINKS_TO_VCF_FILE = "../groundtruth/links_lookup.json"


df = pd.read_csv(GT_AL_FILE)
total_matches_synthesized = df['num_union_cols'].sum() / 2
print(f"Total Matches Synthesized = {total_matches_synthesized}")


with open(GT_LINKS_TO_VCF_FILE, "r") as f:
    links_lookup = json.load(f)

# print(links_lookup)
total_links = 0
for key in links_lookup:
    total_links += len(links_lookup[key])
print(f"Total Links Available = {total_links}")
    


