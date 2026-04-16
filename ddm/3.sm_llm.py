'''
In this file we try out finding schema matching one at a time for
each pairs using llm over the ranked list of ddm.
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
import warnings
import json_repair
from litellm import completion
from openai import OpenAI

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

def get_prompt(key, value, src_file, tgt_file):
    src_codes = pd.read_csv(os.path.join(CODEFILE_PATH, src_file),nrows = 0).columns.to_list()
    src_index = src_codes.index(key)

    preceeding_column_codes = []
    for i in range(src_index):
        preceeding_column_codes.append(src_codes[i])
    
    cand = (
        "Column: "
        + key
        + ", Meaning of the Column: "
        + questions_lookup[key]
    )

    context_cols = [
        "Column: "
        + code
        + ", Meaning of the Column: "
        + questions_lookup[code]
        for code in preceeding_column_codes
    ]
    context = "\n".join(context_cols)
    if not context:
        context = "No Preceeding columns as this is the first column"

    target_cols = [
        "Column: "
        + target_col
        + ", Meaning of the Column: "
        + questions_lookup[target_col]
        for (_, target_col), _ in value.items()
    ]
    targets = "\n".join(target_cols)

    prompt = (
        "Given a Candidate Column from a source table and a list of target columns from the target table, judge the similarity between the candidate and each target column.\n"
        "To provide the context of the candidate column, the name of source table and the list of columns before the candidate column in the source table will also be provided in actual order of their appearance in the table.\n"
        "Return a JSON array of objects, each with 'column' (the target column name) and 'score' (a float between 0 and 1, within two decimal places, 1 means most similar).\n"
        "Do NOT provide any other output text or explanation. Only provide the JSON array.\n"
        "Example:\n"
        "Source Table Name: 'Family Details' \n"
        "Candidate Column: Column: V480015, Meaning of the Column: 'No. of Children?'\n"
        "Columns preceeding candidate column in order of appearance in the source table:\n"
        "Column: V480013, Meaning of the Column: 'Are you the head of your family?'\n"
        "Column: V480014, Meaning of the Column: 'How many wives do you have?'\n"
        "\nTarget Table Name: 'Household Details' \n"
        "Target Columns:\n"
        "Column: V520024, Meaning of the Column: 'Do you have children?'\n"
        "Column: V520025, Meaning of the Column: 'If yes, how many?'\n"
        "Column: V520028, Meaning of the Column: 'Do you rear cattles?'\n"
        "Column: V520029, Meaning of the Column: 'If yes, how many?'\n"
        '\nResponse: [\n {"column": "V520025", "score": 0.98}, \n {"column": "V520024", "score": 0.50}, \n {"column": "V520028", "score": 0.05}, \n {"column": "V520029", "score": 0.05}\n]\n\n'
        "Source Table Name: "
        + src_file[10:-4]
        + "\nCandidate Column: "
        + cand 
        + "\n\nColumns preceeding candidate column in order of appearance in the source table:\n"
        + context
        + "\n\nTarget Table Name: "
        + tgt_file[10:-4]
        + "\nTarget Columns:\n"
        + targets
        + "\n\nResponse: " 
    )
    # print(prompt)
    return prompt
    

def parse_matches(refined_match, tgt_file):
    tgt_cols = pd.read_csv(os.path.join(CODEFILE_PATH, tgt_file), nrows=0).columns.to_list()
    try:
        matches_json = json_repair.loads(refined_match)
        # to check if response has all valid target variable names or some garbage
        for item in matches_json:
            if item['column'] not in tgt_cols:
                warnings.warn(
                    f"Wrong variable name received in response: \nResponse: {matches_json}",
                    UserWarning,
                )
                return None
        matched_columns = []
        for entry in matches_json:
            col_name = entry.get("column")
            score = float(entry.get("score", 0))
            matched_columns.append((col_name, score))
        return matched_columns
    except Exception as e:
        warnings.warn(
            f"Error parsing JSON response: {e}\nRaw response: {refined_match}",
            UserWarning,
        )
        return None

def query_llm(llm_prompt):
    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are an AI trained to perform schema matching by providing column similarity scores.",
        },
        {
            "role": "user",
            "content": llm_prompt
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = messages,
    )

    matches = response.choices[0].message.content
    return matches

def contextful_llm_reranker(data, src_file, tgt_file):
    refined_matches = {}
    for key, value in data.items():
        attempts = 0
        llm_prompt = get_prompt(key, value, src_file, tgt_file)
        while True:
            if attempts >= 5:
                warnings.warn(
                    f"Failed to parse response after 5 attempts. Skipping.",
                    UserWarning,
                )
                refined_match = [
                    v 
                    for (u, v), w in sorted(
                        value.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                ]
                break
            refined_match = query_llm(llm_prompt)
            refined_match = parse_matches(refined_match, tgt_file)
            attempts += 1
            if refined_match is not None:
                break
        refined_matches[key] = {(key, tgt): sim for (tgt,sim) in refined_match}
        # refined_matches[key] = refined_match
    return refined_matches


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
    
    llm_data = contextful_llm_reranker(data, src_fname, tgt_fname)
    # print(llm_data)
    llm_bip_data = {
        (v,w): x
        for _, u in llm_data.items() for (v,w), x in u.items()
    }

    # return data, bip_data
    return llm_data, llm_bip_data, data, bip_data


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
                rr = 1/(i+1)
                rr_sum += rr
                break
    
    return (rr_sum/len(gt_pair_list))


if __name__ == "__main__":

    DIM                                 = 384
    K_NEAREST                           = 50
    TORCH_DEVICE                        = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = TORCH_DEVICE)

    DATAFILES_PATH = "../LoStBench_qn/schema_matching/"
    GT_FILE = "../groundtruth/groundtruth_sm_test_case.csv"
    OUTPUT_STATS_PATH = f"../experiments/DDM/sm_results/ddm_llm_sm_lostbench_qn.csv"
    OUTPUT_PAIRS_PATH = f"../experiments/DDM/sm_results/ddm_llm_sm_lostbench_qn_match_output.csv"
    # RETRIES_STATS_PATH = f"../experiments/DDM/DDM_LLM_retries_top{K_NEAREST}.csv"
    THRESHOLD = 0.4

    CODEFILE_PATH = "../LoStBench/schema_matching/"
    LABELS_LOOKUP_FILE = "../groundtruth/labels_lookup.json"
    QUESTIONS_LOOKUP_FILE = "../groundtruth/questions_lookup.json"
    
    with open(LABELS_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        labels_lookup = json.load(f)
    
    with open(QUESTIONS_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        questions_lookup = json.load(f)

    # retries_columns = [
    #     "pair#",
    #     "num_retries"
    # ]
    # retries_df = pd.DataFrame(columns=retries_columns)
    # retries_df.to_csv(RETRIES_STATS_PATH, index=False)

    columns_output_pairs = [
        "schema1",
        "schema2",
        "match_pairs",
        "ranked_list"
    ]
    out_matches_df = pd.DataFrame(columns=columns_output_pairs)
    out_matches_df.to_csv(OUTPUT_PAIRS_PATH, index=False)

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
        "One2One_RecallAtSizeofGroundTruth",
        "method"
        ]

    out_df = pd.DataFrame(columns=columns)
    out_df.to_csv(OUTPUT_STATS_PATH, index=False) # Save to CSV (headers only)

    gt_df = pd.read_csv(GT_FILE)
    gt_df["match_pairs"] = gt_df["match_pairs"].apply(ast.literal_eval)

    count = 0
    retries = {}
    for _, row in gt_df.iterrows():
        start_time = time.time()

        src_file = os.path.join(DATAFILES_PATH, row["schema1"])
        tgt_file = os.path.join(DATAFILES_PATH, row["schema2"])
        gt_match_pairs = row["match_pairs"]

        concatenated_labels, concatenated_list, q_cols, tgt_cols, n_srows, n_tgtrows = createLists(src_file, tgt_file)
        similarities, similarities_ids = similaritySearch()
        retry = 0
        # while True:
        llm_data, llm_bip_data, data, bip_data = generateBipartiteGraph(q_cols, row["schema1"], row["schema2"])
        sorted_bip_data = [
            (u, v)
            for (u, v), w in sorted(
                bip_data.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]
        sorted_llm_bip_data = [
            (u, v)
            for (u, v), w in sorted(
                llm_bip_data.items(),
                key=lambda item: item[1],
                reverse=True
            )
        ]

        one2one_matching = greedyBipartiteMatching(bip_data)
        one2one_matching_llm = greedyBipartiteMatching(llm_bip_data)
        
        ranked_list = {
                        key: [
                            pair
                            for pair, w in sorted(inner.items(), key=lambda x: x[1], reverse=True)
                        ]
                        for key, inner in data.items()
                    }
        
        llm_ranked_list = {
                        key: [
                            pair
                            for pair, w in sorted(inner.items(), key=lambda x: x[1], reverse=True)
                        ]
                        for key, inner in llm_data.items()
                    }
        
            # if (mrr(gt_match_pairs, llm_data) >= mrr(gt_match_pairs, data)) and \
            #     (recall_at_gt(gt_match_pairs, sorted_llm_bip_data[:len(gt_match_pairs)]) >= recall_at_gt(gt_match_pairs, sorted_bip_data[:len(gt_match_pairs)])) and \
            #     (one2one_recall_at_gt(gt_match_pairs, one2one_matching_llm[:len(gt_match_pairs)]) >= one2one_recall_at_gt(gt_match_pairs, one2one_matching[:len(gt_match_pairs)])):
            #     break
            # else:
            #     retry += 1

            # if retry > 10:
            #     break

        # retry_row = [
        #     f"{count}",
        #     f"{retry}"
        # ]
        # with open(RETRIES_STATS_PATH, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(retry_row)


        out_matches_row = [
            f"{row['schema1']}",
            f"{row['schema2']}",
            f"{one2one_matching_llm}",
            f"{llm_ranked_list}"
        ]

        with open(OUTPUT_PAIRS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_matches_row)
        
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
            f"{mrr(gt_match_pairs, llm_data)}",
            f"{recall_at_gt(gt_match_pairs, sorted_llm_bip_data[:len(gt_match_pairs)])}",
            f"{one2one_recall_at_gt(gt_match_pairs, one2one_matching_llm[:len(gt_match_pairs)])}",
            f"MagnetoGPT"
        ]

        with open(OUTPUT_STATS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_stats_row)
        
        print(f"Completed for {count+1} pairs! out of 450")
        count += 1
        # break #for testing one pair

    # out_matches_df["match_pairs"] = out_matches_df["match_pairs"].apply(json.dumps)
    # out_matches_df.to_csv(OUTPUT_PAIRS_PATH, index=False)
    # print(f"Saved matches successfully to {OUTPUT_PAIRS_PATH}")

