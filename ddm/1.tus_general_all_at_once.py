'''
In this file we try out finding unionable tables all at once
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

def createLists():
    
    concatenated_list = []
    concatenated_labels = []
    query_labels = []
    num_query_cols = 0
    
    for _qfile in _queryFiles:
        data[_qfile] = {}
        
        q_df = pd.read_csv(os.path.join(QUERY_PATH, _qfile))
        q_cols = list(q_df.columns)
        concatenated_list.extend(q_cols)

        src_lbl_lst = ["S"] * len(q_cols)
        concatenated_labels.extend(src_lbl_lst)
        column_counts[_qfile] = len(q_cols)
        num_query_cols += len(q_cols)
        query_labels.extend([_qfile] * len(q_cols))
    
    for _cfile in _candidateFiles:
        c_df = pd.read_csv(os.path.join(DATALAKE_PATH, _cfile))
        c_cols = list(c_df.columns)
        concatenated_list.extend(c_cols)

        cand_lbl_lst = [_cfile] * len(c_cols)
        concatenated_labels.extend(cand_lbl_lst)
        column_counts[_cfile] = len(c_cols)
    
    return concatenated_labels, concatenated_list, query_labels, num_query_cols


def similaritySearch(method="naive"):
    
    emb_time_start = time.time()
    
    TORCH_DEVICE = 'cpu'  # change to 'cuda' if needed
    DIM = 384
    K_NEAREST = 70
    
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=TORCH_DEVICE)
    attributes_embeddings = encoder.encode(concatenated_list, device=TORCH_DEVICE)
    
    emb_time_end = time.time()
    print(f"--------Embedding time: {emb_time_end - emb_time_start:.4f} seconds-------")
    
    idx_time_start = time.time()
    
    series = pd.Series(concatenated_list)
    db_ids = np.array(series.index.values, dtype=np.int64)
    db_vectors = attributes_embeddings.copy().astype(np.float32)
    
    # Normalize ONLY for cosine-based methods
    if method.lower() != "lsh":
        faiss.normalize_L2(db_vectors)
    
    # Create index
    if method.lower() == "naive":
        index = faiss.IndexFlatIP(DIM)
    
    elif method.lower() == "hnsw":
        M = 32
        index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
    
    elif method.lower() == "lsh":
        n_bits = 128
        index = faiss.IndexLSH(DIM, n_bits)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'naive', 'hnsw', or 'lsh'")
    
    # Add vectors
    if method.lower() != "lsh":
        index = faiss.IndexIDMap(index)
        index.add_with_ids(db_vectors, db_ids)
    else:
        index.add(db_vectors)  # LSH does not support IDMap
    
    search_query = attributes_embeddings.copy().astype(np.float32)
    
    if method.lower() != "lsh":
        faiss.normalize_L2(search_query)
    
    idx_time_end = time.time()
    print(f"--------Indexing time: {idx_time_end - idx_time_start:.4f} seconds-------")
    
    srch_time_start = time.time()
    
    similarities, similarities_ids = index.search(search_query, k=K_NEAREST)
    
    srch_time_end = time.time()
    print(f"--------Searching time ({method.upper()}): {srch_time_end - srch_time_start:.4f} seconds-------")
    
    if method.lower() == "lsh":
        similarities = similarities
    else:
        similarities = np.around(np.clip(similarities, 0, 1), decimals=4)
    
    
    return similarities, similarities_ids


def generateBipartiteGraph():
    for i in range(num_query_cols):
        for j in range(similarities.shape[1]):
            if similarities[i][j] < THRESHOLD:
                break
            if similarities_ids[i][j] == i:
                continue
            if concatenated_labels[i] == concatenated_labels[similarities_ids[i][j]]:
                continue
            cand_file = concatenated_labels[similarities_ids[i][j]]
            query_file = query_labels[i]
            if cand_file not in data[query_file]:
                data[query_file][cand_file] = {}
            data[query_file][cand_file][(i, int(similarities_ids[i][j]))] = float(similarities[i][j])



def greedyBipartiteMatching(similarity_dict):
    """
    similarity_dict: dict of (left, right) -> weight
    """

    #Sort edges by descending weight
    edges = sorted(
        [(u, v, w) for (u, v), w in similarity_dict.items()],
        key=lambda x: x[2],
        reverse=True
    )

    matched_left = set()
    matched_right = set()
    matching = {}

    #Greedily pick best edges
    for u, v, w in edges:
        if u not in matched_left and v not in matched_right:
            matching[u] = (v, w)
            matched_left.add(u)
            matched_right.add(v)

    return matching


def getTusResult(tus_result):
    cnt = 0
    for key, value in data.items():
        tus_result[key] = {}
        for k in value:
            result = greedyBipartiteMatching(value[k])
            total_score = 0
            if result:
                # print(f"{key}  ->  {k}")
                for q, (c, score) in result.items():
                    # print(f"{q}({concatenated_list[q]})  ->  {c}({concatenated_list[c]})   (score = {score})")
                    total_score += score
                tus_result[key][k] = total_score/min(column_counts[key], column_counts[k])
                # print("_______________________")
        cnt += 1
        if cnt % 10 == 0:
            print(f"Finished {cnt} out of 115")
    
    tus_result = sortTusResultByScore()
    return tus_result
    

def sortTusResultByScore() -> dict:
    sorted_result = {}

    for q, c in tus_result.items():
        sorted_cands = sorted(
            c.items(),    
            key=lambda x: x[1],  
            reverse=True
        )

        sorted_result[q] = [cand for cand, score in sorted_cands]

    return sorted_result


if __name__ == "__main__":

    QUERY_PATH = "../LoStBench_qn/query/"
    DATALAKE_PATH = "../LoStBench_qn/datalake/"
    OUTPUT_PATH = "../experiments/DDM/tus_results/"
    THRESHOLD = 0.7
    start = time.time()

    _queryFiles = [
        f for f in os.listdir(QUERY_PATH)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(QUERY_PATH, f))
    ]
    _candidateFiles = [
        f for f in os.listdir(DATALAKE_PATH)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(DATALAKE_PATH, f))
    ]

    data = {}
    column_counts = {}
    tus_result = {}
    num_query_cols = 0

    concatenated_labels, concatenated_list, query_labels, num_query_cols = createLists()
    print("lists created")
    print("Starting Similarity Search")
    similarities, similarities_ids = similaritySearch()
    print("Similarity Search Done")
    generateBipartiteGraph()
    print("Bipartite Graphs Generated")

    tus_result = getTusResult(tus_result)

    output_file = os.path.join(OUTPUT_PATH, "tus_result_qn_hnsw.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tus_result, f, indent=2)
        print("Result saved successfully")

    end = time.time()
    print(end-start, "seconds")
