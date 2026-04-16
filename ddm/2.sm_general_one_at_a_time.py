'''
In this file we try out finding schema matching one at a time for
each pairs using ddm.
'''

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
faiss.omp_set_num_threads(1)
import torch
import time
import json
import ast
import csv

def createLists(src_file, tgt_file):
    
    concatenated_list = []
    concatenated_labels = []
    
    src_cols = pd.read_csv(src_file, nrows=0).columns.to_list()
    concatenated_list.extend(src_cols)
    src_lbl_lst = ["S"] * len(src_cols)
    concatenated_labels.extend(src_lbl_lst)

    tgt_cols = pd.read_csv(tgt_file, nrows=0).columns.to_list()
    concatenated_list.extend(tgt_cols)
    tgt_lbl_lst = ["T"] * len(tgt_cols)
    concatenated_labels.extend(tgt_lbl_lst)

    return concatenated_labels, concatenated_list, src_cols, tgt_cols, len(pd.read_csv(src_file)), len(pd.read_csv(tgt_file))


def similaritySearch():
    attributes_embeddings               = encoder.encode(concatenated_list,device = TORCH_DEVICE)

    series                              = pd.Series(concatenated_list)
    index_values                        = series.index.values
    db_vectors                          = attributes_embeddings.copy().astype(np.float32)
    db_ids                              = np.array(index_values, dtype=np.int64)

    faiss.normalize_L2(db_vectors)
    index                               = faiss.IndexFlatIP(DIM)
    index                               = faiss.IndexIDMap(index)
    index.add_with_ids(db_vectors,db_ids)

    search_query                        = attributes_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(search_query)

    similarities, similarities_ids      = index.search(search_query, k=K_NEAREST)
    similarities                        = np.around(np.clip(similarities,0,1),decimals = 4)

    return similarities, similarities_ids


def generateBipartiteGraph(q_cols, src_fname, tgt_fname):
    data = {}
    bip_data = {}
    LOSTBENCH_CODED_PATH = "../LoStBench/schema_matching"
    src_cols = pd.read_csv(os.path.join(LOSTBENCH_CODED_PATH, src_fname), nrows=0).columns.to_list()
    tgt_cols = pd.read_csv(os.path.join(LOSTBENCH_CODED_PATH, tgt_fname), nrows=0).columns.to_list()
    
    for i in range(len(q_cols)):
        data[src_cols[i]] = {}
        for j in range(similarities.shape[1]):
            if similarities[i][j] < THRESHOLD:
                break
            if similarities_ids[i][j] == i:
                continue
            if concatenated_labels[i] == concatenated_labels[similarities_ids[i][j]]:
                continue
            
            bip_data[(src_cols[i], tgt_cols[int(similarities_ids[i][j]) - len(src_cols)])] = float(similarities[i][j])
            data[src_cols[i]][(src_cols[i], tgt_cols[int(similarities_ids[i][j]) - len(src_cols)])] = float(similarities[i][j])
    return data, bip_data


def greedyBipartiteMatching(similarity_dict):
    """
    similarity_dict: dict of (left, right) -> weight
    """

    # Sort edges by descending weight
    edges = sorted(
        [(u, v, w) for (u, v), w in similarity_dict.items()],
        key=lambda x: x[2],
        reverse=True
    )

    matched_left = set()
    matched_right = set()
    matching = []

    # Greedily pick best edges
    for u, v, w in edges:
        if u not in matched_left and v not in matched_right:
            matching.append([u, v])
            matched_left.add(u)
            matched_right.add(v)

    return matching

def one2one_recall_at_gt(gt_match_pairs, one2one_matching):

    match_set = set(tuple(pair) for pair in one2one_matching)
    gt_set = set(tuple(pair) for pair in gt_match_pairs)
    intersection = match_set.intersection(gt_set)

    return len(intersection)/len(gt_match_pairs)


def recall_at_gt(gt_match_pairs, matching):

    match_set = set(tuple(pair) for pair in matching)
    gt_set = set(tuple(pair) for pair in gt_match_pairs)
    intersection = match_set.intersection(gt_set)

    return len(intersection)/len(gt_match_pairs)

def mrr(gt_match_pairs, matchings):

    gt_pair_list =[tuple(pair) for pair in gt_match_pairs]

    sorted_matchings = {
        key: [
            pair
            for pair, w in sorted(inner.items(), key=lambda x: x[1], reverse=True)
        ]
        for key, inner in matchings.items()
    }

    rr_sum = 0
    for key, lst in sorted_matchings.items():
        for i, match in enumerate(lst):
            if match in gt_pair_list:
                # print(match)
                rr = 1/(i+1)
                rr_sum += rr
                break
    
    return (rr_sum/len(gt_pair_list))


if __name__ == "__main__":

    DIM                                 = 384
    K_NEAREST                           = 50
    TORCH_DEVICE                        = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = TORCH_DEVICE)

    DATAFILES_PATH = "../LoStBench_lbl/schema_matching/"
    GT_FILE = "../groundtruth/groundtruth_sm_test_case.csv"
    OUTPUT_STATS_PATH = "../experiments/DDM/ddm_sm_lostbench_lbl.csv"
    OUTPUT_PAIRS_PATH = "../experiments/DDM/ddm_sm_lostbench_lbl_match_output.csv"
    THRESHOLD = 0.4

    columns_output_pairs = [
        "schema1",
        "schema2",
        "match_pairs",
        "ranked_list"
    ]
    out_matches_df = pd.DataFrame(columns=columns_output_pairs)

    columns = [
        "source_table", 
        "target_table",
        "ncols_src",
        "ncols_tgt",
        "nrows_src",
        "nrows_tgt",
        "nmatches",
        "runtime",
        "mrr",
        "All_RecallAtSizeofGroundTruth",
        "One2One_RecallAtSizeofGroundTruth"
        ]

    out_df = pd.DataFrame(columns=columns)
    out_df.to_csv(OUTPUT_STATS_PATH, index=False) # Save to CSV (headers only)

    gt_df = pd.read_csv(GT_FILE)
    gt_df["match_pairs"] = gt_df["match_pairs"].apply(ast.literal_eval)

    count = 0
    for _, row in gt_df.iterrows():
        start_time = time.time()

        src_file = os.path.join(DATAFILES_PATH, row["schema1"])
        tgt_file = os.path.join(DATAFILES_PATH, row["schema2"])
        gt_match_pairs = row["match_pairs"]

        concatenated_labels, concatenated_list, q_cols, tgt_cols, n_srows, n_tgtrows = createLists(src_file, tgt_file)
        similarities, similarities_ids = similaritySearch()
        data, bip_data = generateBipartiteGraph(q_cols, row["schema1"], row["schema2"])
        sorted_bip_data = [
            (u, v)
            for (u, v), w in sorted(
                bip_data.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
        one2one_matching = greedyBipartiteMatching(bip_data)
        # print(one2one_matching)
        out_matches_row = pd.DataFrame([{
            'schema1': row['schema1'],
            'schema2': row['schema2'],
            'match_pairs': one2one_matching,
            "ranked_list": {
                                key: [
                                    pair
                                    for pair, w in sorted(inner.items(), key=lambda x: x[1], reverse=True)
                                ]
                                for key, inner in data.items()
                            }
        }])

        out_matches_df = pd.concat([out_matches_df, out_matches_row], ignore_index=True)
        
        end_time = time.time()
        runtime = end_time - start_time

        out_stats_row = [
            f"{row['schema1']}",
            f"{row['schema2']}",
            f"{len(q_cols)}",
            f"{len(tgt_cols)}",
            f"{n_srows}",
            f"{n_tgtrows}",
            f"{len(gt_match_pairs)}",
            f"{runtime}",
            f"{mrr(gt_match_pairs, data)}",
            f"{recall_at_gt(gt_match_pairs, sorted_bip_data[:len(gt_match_pairs)])}",
            f"{one2one_recall_at_gt(gt_match_pairs, one2one_matching[:len(gt_match_pairs)])}"
        ]

        with open(OUTPUT_STATS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_stats_row)
        
        if count % 50 == 0:
            print(f"Completed for {count+1} pairs! out of 450")
        count += 1

    out_matches_df["match_pairs"] = out_matches_df["match_pairs"].apply(json.dumps)
    out_matches_df.to_csv(OUTPUT_PAIRS_PATH, index=False)
    print(f"Saved matches successfully to {OUTPUT_PAIRS_PATH}")

