import pandas as pd
import os
import statistics
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_name_overlap(src_cols, tgt_cols):
    common = set(src_cols).intersection(tgt_cols)
    return len(common)/min(len(src_cols), len(tgt_cols))


def get_value_overlap(src_df, tgt_df):

    big_list_src = [
            x for x in pd.unique(src_df.select_dtypes(exclude="number").values.ravel())
            if pd.notna(x)
        ]
    
    big_list_tgt = [
            x for x in pd.unique(tgt_df.select_dtypes(exclude="number").values.ravel())
            if pd.notna(x)
        ]
    
    common = set(big_list_src).intersection(set(big_list_tgt))

    if min(len(set(big_list_src)), len(set(big_list_tgt))) == 0:
        return 0
    
    return (len(common)/ min(len(set(big_list_src)), len(set(big_list_tgt))))

def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return ast.literal_eval(x)  # safely converts string "[...]" to list


if __name__ == "__main__":

    OUTPUT_STATS_PATH = "sm_overlap_stats.csv"
    # columns_output = [
    #         "benchmark",
    #         "name_overlaps",
    #         "value_overlaps",
    #     ]
    # out_df = pd.DataFrame(columns=columns_output)
    # out_df.to_csv(OUTPUT_STATS_PATH, index=False)

    '''
    ------ LoStBench Stats in General ----------------
    '''
    
    # FILE_FOR_DATASET_STATS = "../experiments/magneto/ddm_sm_lostbench_qn.csv"
    # df = pd.read_csv(FILE_FOR_DATASET_STATS)

    # min_col_num = min(df['ncols_src'].min(), df['ncols_tgt'].min())
    # max_col_num = max(df['ncols_src'].max(), df['ncols_tgt'].max())
    # print(f"Num Columns in LoStBench SM: {min_col_num}-{max_col_num}")

    # min_row_num = min(df['nrows_src'].min(), df['nrows_tgt'].min())
    # max_row_num = max(df['nrows_src'].max(), df['nrows_tgt'].max())
    # print(f"Num Rows in LoStBench SM: {min_row_num}-{max_row_num}")

    '''
    ------ LoStBench Overlap Stats ----------------
    '''
    # SOURCE_FOLDER = "../LoStBench_lbl/schema_matching/"
    # GT_FILE = "../groundtruth/groundtruth_sm_test_case.csv"

    # name_overlaps = []
    # value_overlaps = []
    
    # gt_df = pd.read_csv(GT_FILE)
    # for _, row in gt_df.iterrows():
    #     src_file = os.path.join(SOURCE_FOLDER, row["schema1"])
    #     tgt_file = os.path.join(SOURCE_FOLDER, row["schema2"])

    #     src_df = pd.read_csv(src_file)
    #     src_cols = src_df.columns.to_list()

    #     tgt_df = pd.read_csv(tgt_file)
    #     tgt_cols = tgt_df.columns.to_list()

    #     name_overlap = get_name_overlap(src_cols, tgt_cols)
    #     name_overlaps.append(name_overlap)

    #     value_overlap = get_value_overlap(src_df, tgt_df)
    #     value_overlaps.append(value_overlap)
    
    # print(f"Name Overlap for LoStBench: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for LoStBench: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"LoStBench",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    '''
    ------GDC Name/Value Overlap Stats -------
    '''
    # FOLDER = "../other_sm_benchmark_datasets/gdc/ground-truth/"

    # name_overlaps = []

    # files = [file for file in os.listdir(FOLDER) if file.endswith('.csv')]
    # for file in files:
    #     df = pd.read_csv(os.path.join(FOLDER, file))
    #     src = df['original_paper_variable_names'].to_list()
    #     tgt = df ['GDC_format_variable_names'].to_list()
    #     name_overlap = get_name_overlap(src, tgt)
    #     name_overlaps.append(name_overlap)
    
    # print(f"Name Overlap for GDC: {statistics.mean(name_overlaps)}")

    
    # GT_FOLDER = "../other_sm_benchmark_datasets/gdc/ground-truth/"
    # SRC_FOLDER = "../other_sm_benchmark_datasets/gdc/source-tables/"
    # TGT_FILE = "../other_sm_benchmark_datasets/gdc/target-tables/gdc_unique_columns_concat_values.csv"

    # value_overlaps = []

    # files = [file for file in os.listdir(GT_FOLDER) if file.endswith('.csv')]
    # for file in files:
    #     gt_df = pd.read_csv(os.path.join(GT_FOLDER, file))

    #     tgt = gt_df ['GDC_format_variable_names'].to_list()
    #     tgt = list(set(tgt)) # to remove duplicates

    #     src_df = pd.read_csv(os.path.join(SRC_FOLDER, file))
    #     tgt_df = pd.read_csv(TGT_FILE)
    #     tgt_df = tgt_df[[c for c in tgt if c in tgt_df.columns]]
        
    #     value_overlap = get_value_overlap(src_df, tgt_df)
    #     value_overlaps.append(value_overlap)

    # print(f"Value Overlap for GDC: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"GDC",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    '''
    ------Magellan Name/Value Overlap Stats -------
    '''
    # SRC_FOLDER = "../other_sm_benchmark_datasets/Valentine-datasets/Magellan/"
    # testcase_folders = os.listdir(SRC_FOLDER)

    # name_overlaps = []
    # value_overlaps = []

    # for case in testcase_folders:
    #     src_file = f"{case}_source.csv"
    #     tgt_file = f"{case}_target.csv"

    #     src_df = pd.read_csv(os.path.join(SRC_FOLDER, case, src_file))
    #     src_cols = src_df.columns.to_list()

    #     tgt_df = pd.read_csv(os.path.join(SRC_FOLDER, case, tgt_file))
    #     tgt_cols = tgt_df.columns.to_list()

    #     name_overlap = get_name_overlap(src_cols, tgt_cols)
    #     name_overlaps.append(name_overlap)

    #     value_overlap = get_value_overlap(src_df, tgt_df)
    #     value_overlaps.append(value_overlap)

    # print(f"Name Overlap for Magellan: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for Magellan: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"Magellan",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    '''
    ------Wikidata Name/Value Overlap Stats -------
    '''
    # SRC_FOLDER = "../other_sm_benchmark_datasets/Valentine-datasets/Wikidata/Musicians/"
    # testcase_folders = os.listdir(SRC_FOLDER)

    # name_overlaps = []
    # value_overlaps = []

    # for case in testcase_folders:
    #     src_file = f"{case.lower()}_source.csv"
    #     tgt_file = f"{case.lower()}_target.csv"

    #     src_df = pd.read_csv(os.path.join(SRC_FOLDER, case, src_file))
    #     src_cols = src_df.columns.to_list()

    #     tgt_df = pd.read_csv(os.path.join(SRC_FOLDER, case, tgt_file))
    #     tgt_cols = tgt_df.columns.to_list()

    #     name_overlap = get_name_overlap(src_cols, tgt_cols)
    #     name_overlaps.append(name_overlap)

    #     value_overlap = get_value_overlap(src_df, tgt_df)
    #     value_overlaps.append(value_overlap)

    # print(f"Name Overlap for Wikidata: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for Wikidata: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"Wikidata",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    '''
    ------OpenData Name/Value Overlap Stats -------
    '''
    # SRC_FOLDER = "../other_sm_benchmark_datasets/Valentine-datasets/OpenData/"
    # valentine_match_types = os.listdir(SRC_FOLDER)

    # name_overlaps = []
    # value_overlaps = []

    # for val_match_type in valentine_match_types:
    #     testcase_folders = os.listdir(os.path.join(SRC_FOLDER,val_match_type))
    
    #     for case in testcase_folders:
    #         src_file = f"{case.lower()}_source.csv"
    #         tgt_file = f"{case.lower()}_target.csv"

    #         src_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, src_file))
    #         src_cols = src_df.columns.to_list()

    #         tgt_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, tgt_file))
    #         tgt_cols = tgt_df.columns.to_list()

    #         name_overlap = get_name_overlap(src_cols, tgt_cols)
    #         name_overlaps.append(name_overlap)

    #         value_overlap = get_value_overlap(src_df, tgt_df)
    #         value_overlaps.append(value_overlap)

    # print(f"Name Overlap for OpenData: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for OpenData: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"OpenData",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    '''
    ------ChEMBL Name/Value Overlap Stats -------
    '''
    # SRC_FOLDER = "../other_sm_benchmark_datasets/Valentine-datasets/ChEMBL/"
    # valentine_match_types = os.listdir(SRC_FOLDER)

    # name_overlaps = []
    # value_overlaps = []

    # for val_match_type in valentine_match_types:
    #     testcase_folders = os.listdir(os.path.join(SRC_FOLDER,val_match_type))
    
    #     for case in testcase_folders:
    #         src_file = f"{case.lower()}_source.csv"
    #         tgt_file = f"{case.lower()}_target.csv"

    #         src_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, src_file))
    #         src_cols = src_df.columns.to_list()

    #         tgt_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, tgt_file))
    #         tgt_cols = tgt_df.columns.to_list()

    #         name_overlap = get_name_overlap(src_cols, tgt_cols)
    #         name_overlaps.append(name_overlap)

    #         value_overlap = get_value_overlap(src_df, tgt_df)
    #         value_overlaps.append(value_overlap)

    # print(f"Name Overlap for ChEMBL: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for ChEMBL: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"ChEMBL",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)
    

    '''
    ------TPC-DI Name/Value Overlap Stats -------
    '''
    # SRC_FOLDER = "../other_sm_benchmark_datasets/Valentine-datasets/TPC-DI/"
    # valentine_match_types = os.listdir(SRC_FOLDER)

    # name_overlaps = []
    # value_overlaps = []

    # for val_match_type in valentine_match_types:
    #     testcase_folders = os.listdir(os.path.join(SRC_FOLDER,val_match_type))
    
    #     for case in testcase_folders:
    #         src_file = f"{case.lower()}_source.csv"
    #         tgt_file = f"{case.lower()}_target.csv"

    #         src_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, src_file))
    #         src_cols = src_df.columns.to_list()

    #         tgt_df = pd.read_csv(os.path.join(SRC_FOLDER, val_match_type, case, tgt_file))
    #         tgt_cols = tgt_df.columns.to_list()

    #         name_overlap = get_name_overlap(src_cols, tgt_cols)
    #         name_overlaps.append(name_overlap)

    #         value_overlap = get_value_overlap(src_df, tgt_df)
    #         value_overlaps.append(value_overlap)

    # print(f"Name Overlap for TPC-DI: {statistics.mean(name_overlaps)}")
    # print(f"Value Overlap for TPC-DI: {statistics.mean(value_overlaps)}")

    # out_stats_row = [
    #     f"TPC-DI",
    #     f"{name_overlaps}",
    #     f"{value_overlaps}",
    # ]

    # with open(OUTPUT_STATS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_stats_row)

    output_stats = pd.read_csv(OUTPUT_STATS_PATH)
    results = []

    for _, row in output_stats.iterrows():
        name_vals = parse_list(row['name_overlaps'])
        value_vals = parse_list(row['value_overlaps'])
        
        name_mean = np.mean(name_vals) if name_vals else np.nan
        name_var = np.var(name_vals) if name_vals else np.nan
        
        value_mean = np.mean(value_vals) if value_vals else np.nan
        value_var = np.var(value_vals) if value_vals else np.nan
        
        results.append({
            'benchmark': row['benchmark'],
            'name_mean': name_mean,
            'name_variance': name_var,
            'value_mean': value_mean,
            'value_variance': value_var
        })

    result_df = pd.DataFrame(results)

    # Print
    for _, row in result_df.iterrows():
        print(f"{row['benchmark']} -> "
            f"name(mean={row['name_mean']:.4f}, var={row['name_variance']:.4f}), "
            f"value(mean={row['value_mean']:.4f}, var={row['value_variance']:.4f})")

    '''
    ------ Plot Stats Distribution ------
    '''

    # sns.set_theme(style="whitegrid", context="talk", palette="colorblind")

    # df = pd.read_csv(OUTPUT_STATS_PATH)
    # df["name_overlaps"] = df["name_overlaps"].apply(ast.literal_eval)
    # df["value_overlaps"] = df["value_overlaps"].apply(ast.literal_eval)
    # benchmarks = df["benchmark"].unique()

    # name_rows = []
    # value_rows = []

    # for _, row in df.iterrows():
    #     for v in row["name_overlaps"]:
    #         name_rows.append({"benchmark": row["benchmark"], "overlap": v})
    #     for v in row["value_overlaps"]:
    #         value_rows.append({"benchmark": row["benchmark"], "overlap": v})

    # name_df = pd.DataFrame(name_rows)
    # value_df = pd.DataFrame(value_rows)

    # fig, axes = plt.subplots(2, 2, figsize=(16,10))

    # # cdf
    # for b in benchmarks:
    #     row = df[df["benchmark"] == b].iloc[0]

    #     sns.ecdfplot(
    #         x=row["name_overlaps"],
    #         ax=axes[0,0],
    #         linewidth=3,
    #         label=b
    #     )

    #     sns.ecdfplot(
    #         x=row["value_overlaps"],
    #         ax=axes[0,1],
    #         linewidth=3,
    #         label=b
    #     )

    # axes[0,0].set_title("CDF of Name Overlaps", fontsize=18)
    # axes[0,1].set_title("CDF of Value Overlaps", fontsize=18)

    # axes[0,0].set_xlabel("Overlap", fontsize=16)
    # axes[0,1].set_xlabel("Overlap", fontsize=16)
    # axes[0,0].set_ylabel("CDF", fontsize=16)

    # axes[0,0].legend(fontsize=12)
    # axes[0,1].legend(fontsize=12)

    # # strip plot
    # sns.stripplot(
    #     data=name_df,
    #     x="overlap",
    #     y="benchmark",
    #     ax=axes[1,0],
    #     size=15,
    #     alpha=0.7,
    #     hue="benchmark",
    #     jitter=False,
    #     dodge=False
    # )

    # sns.stripplot(
    #     data=value_df,
    #     x="overlap",
    #     y="benchmark",
    #     ax=axes[1,1],
    #     size=15,
    #     alpha=0.7,
    #     hue="benchmark",
    #     jitter=False,
    #     dodge=False
    # )

    # axes[1,0].set_title("Name Overlap Distribution", fontsize=18)
    # axes[1,1].set_title("Value Overlap Distribution", fontsize=18)

    # axes[1,0].set_xlabel("Overlap", fontsize=16)
    # axes[1,1].set_xlabel("Overlap", fontsize=16)
    # axes[1,0].set_ylabel("Benchmark", fontsize=16)
    # axes[1,1].set_ylabel("Benchmark", fontsize=16)

    # name_counts = name_df.groupby("benchmark").size()
    # value_counts = value_df.groupby("benchmark").size()

    # y_labels_name = [t.get_text() for t in axes[1,0].get_yticklabels()]
    # y_ticks_name = axes[1,0].get_yticks()
    # y_map_name = dict(zip(y_labels_name, y_ticks_name))

    # y_labels_value = [t.get_text() for t in axes[1,1].get_yticklabels()]
    # y_ticks_value = axes[1,1].get_yticks()
    # y_map_value = dict(zip(y_labels_value, y_ticks_value))

    # # for annotating overlaps
    # for b in benchmarks:
    #     if b in name_counts and b in y_map_name:
    #         axes[1,0].text(
    #             axes[1,0].get_xlim()[1] * 1.02,  # just outside right
    #             y_map_name[b],
    #             f"n={name_counts[b]}",
    #             va="center",
    #             ha="left",
    #             fontsize=12
    #         )

    # for b in benchmarks:
    #     if b in value_counts and b in y_map_value:
    #         axes[1,1].text(
    #             axes[1,1].get_xlim()[1] * 1.02,
    #             y_map_value[b],
    #             f"n={value_counts[b]}",
    #             va="center",
    #             ha="left",
    #             fontsize=12
    #         )

    # # Adjust layout for annotations
    # plt.tight_layout(rect=[0, 0, 0.85, 1])  #extra room on the right
    # plt.show()



            
            