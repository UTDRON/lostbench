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
from itertools import combinations

def get_pairs_with_singletons(clusters):
    """
    Return pairs for clusters, but also treat singletons as correct units.
    Each pair is a frozenset. Singleton items are represented as frozenset of size 1.
    """
    pairs = set()
    for cluster in clusters:
        if len(cluster) == 1:
            pairs.add(frozenset(cluster))  # singleton counts as a "pair"
        else:
            for pair in combinations(cluster, 2):
                pairs.add(frozenset(pair))
    return pairs

def pairwise_precision_recall_f1_singletons(gt_clusters, pred_clusters):
    """
    Compute precision, recall, F1 considering singletons.
    """
    gt_pairs = get_pairs_with_singletons(gt_clusters)
    pred_pairs = get_pairs_with_singletons(pred_clusters)

    true_positives = gt_pairs & pred_pairs
    precision = len(true_positives) / len(pred_pairs) if pred_pairs else 0
    recall = len(true_positives) / len(gt_pairs) if gt_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def merge_pairs(table1, table2, encoder, TORCH_DEVICE):
    concatenated_list = []
    concatenated_labels = []
    concatenated_codes = []

    for col in table1.columns:
        concatenated_list.append(col.questions[-1])
        concatenated_codes.append(col.codes[-1])
        concatenated_labels.append("S")
    
    for col in table2.columns:
        concatenated_list.append(col.questions[-1])
        concatenated_codes.append(col.codes[-1])
        concatenated_labels.append("T")
    
    src_cols = [col.codes[-1] for col in table1.columns]
    tgt_cols = [col.codes[-1] for col in table2.columns]
    similarities, similarities_ids = similaritySearch(encoder, concatenated_list, TORCH_DEVICE, K_NEAREST = len(concatenated_list))
    data, bip_data = generateBipartiteGraph(similarities, similarities_ids, concatenated_labels, src_cols, tgt_cols)
    one2one_matching = greedyBipartiteMatching(bip_data)

    return data, one2one_matching


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


def generateBipartiteGraph(similarities, similarities_ids, concatenated_labels, src_cols, tgt_cols):
    data = {}
    bip_data = {}
    THRESHOLD = 0.05
    
    for i in range(len(src_cols)):
        data[src_cols[i]] = []
        for j in range(similarities.shape[1]):
            if similarities[i][j] < THRESHOLD:
                break
            if similarities_ids[i][j] == i:
                continue
            if concatenated_labels[i] == concatenated_labels[similarities_ids[i][j]]:
                continue
            
            bip_data[(src_cols[i], tgt_cols[int(similarities_ids[i][j]) - len(src_cols)])] = float(similarities[i][j])
            data[src_cols[i]].append(tgt_cols[int(similarities_ids[i][j]) - len(src_cols)])
    return data, bip_data


def similaritySearch(encoder, concatenated_list, TORCH_DEVICE, K_NEAREST, DIM = 384):
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


def max_rank_for_ideal_recall_pairs(gt, var, ranked_list, tgt_cols):
    for subgraph in gt:
        if var in subgraph:
            # print(var," ----> ", ranked_list )
            for i,cand in enumerate(ranked_list):
                if cand in subgraph:
                    return i+1
            
            if set(tgt_cols).intersection(set(subgraph)):    
                return np.inf


def max_rank_for_ideal_recall(graphs, var, ranked_list):

    if ranked_list and graphs:
        for subgraph in graphs:
            if var in subgraph:
                num_relevant = 0
                num_relevant_at_gt = 0
                gt_num = len(subgraph) - 1
                for i,cand in enumerate(ranked_list):
                    if cand in subgraph:
                        num_relevant += 1
                    if i+1 == gt_num:
                        num_relevant_at_gt = num_relevant
                    
                    if num_relevant == gt_num:
                        break

                if num_relevant == gt_num:
                    return (num_relevant_at_gt, num_relevant, gt_num, i+1, subgraph)

                return (num_relevant_at_gt, num_relevant, gt_num, np.inf, subgraph)


def parse_args(parser):
    parser.add_argument(
        "--hitl",
        choices=["y", "n"],
        required=True,
        help="Human-in-the-loop flag: y or n"
    )

    parser.add_argument(
        "--mode",
        choices=["all", "pair"],
        required=True,
        help="Mode: all-at-once or pairwise"
    )

    args = parser.parse_args()

    hitl = args.hitl
    mode = args.mode

    return hitl, mode
    

def find_components(graph):
    visited = set()
    components = []

    def dfs(node, comp):
        visited.add(node)
        comp.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, comp)

    for node in graph:
        if node not in visited:
            comp = []
            dfs(node, comp)
            components.append(comp)

    return components


def merge_all_at_once(
        key, 
        value, 
        TORCH_DEVICE, 
        encoder,
        DATA_PATH = "../LoStBench_qn/datalake/",
        THRESHOLD = 0.4
    ):

    CODE_DATA_PATH = "../LoStBench/datalake/"
    concatenated_list = []
    concatenated_labels = []
    concatenated_codes = []
    # query_labels = []
    column_counts = {}
    total_candidate_num = 0
    out_ranked_vars = {}
    out_ranked_tgt_files = {}

    src_cols = pd.read_csv(os.path.join(DATA_PATH, key), nrows = 0).columns.to_list()
    concatenated_list.extend(src_cols)
    src_cols = pd.read_csv(os.path.join(CODE_DATA_PATH, key), nrows = 0).columns.to_list()
    concatenated_codes.extend(src_cols)
    src_lbls = ["S"] * len(src_cols)
    column_counts[key] = len(src_cols)
    concatenated_labels.extend(src_lbls)
    total_candidate_num += len(src_cols)

    for tgt_file in value:
        tgt_cols = pd.read_csv(os.path.join(DATA_PATH, tgt_file), nrows = 0).columns.to_list()
        concatenated_list.extend(tgt_cols)
        tgt_cols = pd.read_csv(os.path.join(CODE_DATA_PATH, tgt_file), nrows = 0).columns.to_list()
        concatenated_codes.extend(tgt_cols)
        tgt_lbls = [tgt_file] * len(tgt_cols)
        concatenated_labels.extend(tgt_lbls)
        column_counts[tgt_file] = len(src_cols)
        total_candidate_num += len(tgt_cols)
    
    similarities, similarities_ids = similaritySearch(encoder, concatenated_list, TORCH_DEVICE, K_NEAREST = total_candidate_num)
    
    # for i in range(len(src_cols)):
    scored_tuples = []
    for i in range(similarities.shape[1]):
        out_ranked_vars[concatenated_codes[i]] = []
        out_ranked_tgt_files[concatenated_codes[i]] = []
        for j in range(similarities.shape[1]):
            if similarities[i][j] < THRESHOLD:
                break
            if similarities_ids[i][j] == i:
                continue
            if concatenated_labels[i] == concatenated_labels[similarities_ids[i][j]]:
                continue
            cand_file = concatenated_labels[similarities_ids[i][j]]
            cand_var = concatenated_codes[int(similarities_ids[i][j])] 
            
            out_ranked_vars[concatenated_codes[i]].append(cand_var)
            out_ranked_tgt_files[concatenated_codes[i]].append(cand_file)

            scored_tuples.append(
                (
                    concatenated_codes[i],
                    cand_var,
                    similarities[i][j]
                )
            )
    
    return out_ranked_vars, out_ranked_tgt_files, total_candidate_num, scored_tuples


def process_scored_tuples(scored_tuples):
    scored_tuples = sorted(scored_tuples, key=lambda x: x[2], reverse=True)
    flat_pred = []
    pred_groups = []

    for (a,b,score) in scored_tuples:
        if a not in flat_pred:
            flat_pred.append(a)
            pred_groups.append[[a]]
            
        pass