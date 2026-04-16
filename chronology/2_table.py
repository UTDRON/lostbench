import pandas as pd
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
faiss.omp_set_num_threads(1)
import torch
import argparse
from utils import merge_all_at_once, parse_args, find_components, max_rank_for_ideal_recall, merge_pairs, max_rank_for_ideal_recall_pairs, \
get_pairs_with_singletons, pairwise_precision_recall_f1_singletons, process_scored_tuples
import time
import csv

class Table:
    def __init__(self,
                table_type: str, 
                names: list,
                columns: list,
                gt: list,
                # df: pd.DataFrame,
                sources: list = None
            ):
        
        self.table_type = table_type
        self.names = names
        self.columns = columns
        self.gt = gt
        # self.df = df
        self.sources = [self] if self.table_type == "pure" else sources
        

    def merge_tables(self, new_table, hitl='n'):
        if new_table:

            srch_output, one2one_matching = merge_pairs(self, new_table, encoder, TORCH_DEVICE)
            max_ranks_for_ideal_recall_pairwise = []
            for var, ranked_list in srch_output.items():
                result = max_rank_for_ideal_recall_pairs(self.gt, var, ranked_list, [col.codes[-1] for col in new_table.columns])
                if result:
                    max_ranks_for_ideal_recall_pairwise.append(result)
            
            max_ranks_for_ideal_recall_pairwise = [
                len(new_table.columns) if m_r == np.inf else m_r
                for m_r in max_ranks_for_ideal_recall_pairwise
            ]

            if hitl == 'y':
                merged_cols = []
                for col1 in self.columns:
                    for col2 in new_table.columns:
                        for subgraph in self.gt:
                            if col1.codes[-1] in subgraph and col2.codes[-1] in subgraph:
                                col1.merge_columns(col2)
                                merged_cols.append(col2)

            else:
                merged_cols = []
                for [u,v] in one2one_matching:
                    col1= None
                    col2 = None
                    for col in self.columns:
                        if u == col.codes[-1]:
                            col1 = col

                    for col in new_table.columns:
                        if v == col.codes[-1]:
                            col2 = col
                    
                    col1.merge_columns(col2)
                    merged_cols.append(col2)
            
            self.table_type = "merged"
            self.names.extend(new_table.names)
            self.columns.extend(x for x in new_table.columns if x not in merged_cols)
            self.sources.extend(new_table.sources)
 
            return self, max_ranks_for_ideal_recall_pairwise


class Column:
    def __init__(self, 
                col_type: str,
                codes: list,
                labels: list,
                questions: list,
                linked_vcfs: list,
                sources: list = None
            ):
        
        self.col_type = col_type
        self.codes = codes
        self.labels = labels
        self.questions = questions
        self.linked_vcfs = linked_vcfs
        self.sources = [self] if self.col_type == "pure" else sources
    
    def merge_columns(self, new_column):
        if new_column:
            self.col_type = "merged"
            self.codes.extend(new_column.codes)
            self.labels.extend(new_column.labels)
            self.questions.extend(new_column.questions)
            self.linked_vcfs.extend(new_column.linked_vcfs)
            self.linked_vcfs = list(set(self.linked_vcfs))
            self.sources.extend(new_column.sources)


def createInitialTablesList(test_case: list, adj_list: dict) -> list:
    tables_list = []
    for table in test_case:
        columns = []
        column_codes = pd.read_csv(os.path.join(LOSTBENCH_PATH, table), nrows=0).columns.to_list()
        for col in column_codes:
            columns.append(
                Column(
                    "pure",
                    [col],
                    [label_lookup[col]],
                    [question_lookup[col]],
                    links_lookup[col] if col in links_lookup else []
                )
            )

        tables_list.append(
            Table(
                "pure",
                [table],
                columns,
                find_components(adj_list)
            )
        )
    
    return tables_list


# run python 2_table.py --hitl n or y --mode all or pair
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run simulation script")
    hitl, mode = parse_args(parser)

    DIM                                 = 384
    K_NEAREST                           = 50
    TORCH_DEVICE                        = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder                             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device = TORCH_DEVICE)
    THRESHOLD                           = 0.05

    LOSTBENCH_PATH = "../LoStBench/datalake/"
    LBL_LOOKUP_PATH = "../groundtruth/labels_lookup.json"
    QN_LOOKUP_PATH = "../groundtruth/questions_lookup.json"
    LINKS_LOOKUP_PATH = "../groundtruth/links_lookup.json"
    MATCHES_LOOKUP_PATH = "test_case_combo_lookup.json"
    UNDIRECTED_GRAPH = "undirected_graph.json"
    TEST_CASES_FILE = "test_case_random.json"
    TABLE_COLUMNS_LOOKUP = "../groundtruth/table_columns_lookup.json"
    OUTPUT_STATS_PATH = "all_at_once_all.csv"

    with open(TABLE_COLUMNS_LOOKUP, "r", encoding = "utf-8") as f:
        tbl_col_lookup = json.load(f)

    with open(LBL_LOOKUP_PATH, "r", encoding = "utf-8") as f:
        label_lookup = json.load(f)
    
    with open(QN_LOOKUP_PATH, "r", encoding = "utf-8") as f:
        question_lookup = json.load(f)
    
    with open(LINKS_LOOKUP_PATH, "r", encoding = "utf-8") as f:
        links_lookup = json.load(f)

    with open(MATCHES_LOOKUP_PATH, "r", encoding = "utf-8") as f:
        matches_lookup = json.load(f)

    with open(UNDIRECTED_GRAPH, "r", encoding = "utf-8") as f:
        gt_graphs = json.load(f)
    
    with open(TEST_CASES_FILE, "r", encoding = "utf-8") as f:
        test_cases = json.load(f)

    if mode == 'all':
        data = {}
        data_lookup_files = {}

        columns_output = [
            "test_file",
            "gt_nums",
            "num_relevants@gt",
            "recall@gt",
            "num_relevants",
            "total_recall",
            "max_ranks_4_ideal_recall",
            "time"
        ]
        out_df = pd.DataFrame(columns=columns_output)
        out_df.to_csv(OUTPUT_STATS_PATH, index=False)

        count_files = 0
    
        for key, value in test_cases.items():
            start_time = time.time()
            out_ranked_vars, out_ranked_tgt_files, total_cand_num, scored_tuples = merge_all_at_once(key, value, TORCH_DEVICE, encoder)
            end_time = time.time()
            spent_time = end_time-start_time
            data[key] = out_ranked_vars
            data_lookup_files[key] = out_ranked_tgt_files

            graph_components = find_components(gt_graphs[key])
            num_relevants_at_gts = []
            num_relevants = []
            gt_nums = []
            recalls_at_gts = []
            total_recalls = []
            max_ranks_for_ideal_recall = []
            done_vars = []
            for var, ranked_list in data[key].items():
                if var not in done_vars:  # only size of subgraphs
                    result = max_rank_for_ideal_recall(graph_components, var, ranked_list)
                    if result:
                        num_relevants_at_gts.append(result[0])
                        num_relevants.append(result[1])
                        gt_nums.append(result[2])
                        max_ranks_for_ideal_recall.append(result[3])
                        done_vars.extend(result[4]) # only size of subgraphs
            
            max_ranks_for_ideal_recall = [
                total_cand_num if m_r == np.inf else m_r
                for m_r in max_ranks_for_ideal_recall
            ]

            recalls_at_gts = [a/b for a, b in zip(num_relevants_at_gts, gt_nums)]
            total_recalls = [a/b for a, b in zip(num_relevants, gt_nums)]

            precision, recall, f1 = process_scored_tuples(scored_tuples)

            out_stats_row = [
                f"{key}",
                f"{gt_nums}",
                f"{num_relevants_at_gts}",
                f"{recalls_at_gts}",
                f"{num_relevants}",
                f"{total_recalls}",
                f"{max_ranks_for_ideal_recall}",
                f"{spent_time}"
            ]

            with open(OUTPUT_STATS_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(out_stats_row)

            count_files += 1
            print(f"Finished {count_files}")
            break
        
    else:
        if hitl == 'n':
            OUTPUT_STATS_PATH = "pairwise_no_hitl.csv"
        else:
            OUTPUT_STATS_PATH = "pairwise_hitl_random.csv"

        columns_output = [
            "test_file",
            "max_ranks_4_ideal_recall",
            "time",
            "precision",
            "recall",
            "f1-score"
        ]
        out_df = pd.DataFrame(columns=columns_output)
        out_df.to_csv(OUTPUT_STATS_PATH, index=False)

        count_files = 0

        for key, values in test_cases.items():
            max_ranks_for_ideal_recall_all = []
            start_time = time.time()
            test_case = [key]#.extend(values)
            test_case.extend(values)

            tables_list = createInitialTablesList(test_case, gt_graphs[key])

            graph_components = find_components(gt_graphs[key])
        
            for i in range(len(tables_list) - 1):
                if i == 0:
                    merged_table, max_ranks_for_ideal_recall_pairwise = tables_list[0].merge_tables(tables_list[1], hitl)
                else: 
                    merged_table, max_ranks_for_ideal_recall_pairwise = merged_table.merge_tables(tables_list[i+1], hitl)
                    # break
                max_ranks_for_ideal_recall_all.append(max_ranks_for_ideal_recall_pairwise)

            end_time = time.time()

            # --- for accuracy measures evaluation (precision, recall, f1 scores) ---
            predicted_groups = []
            for col in merged_table.columns:
                predicted_groups.append(col.codes)

            all_vars = [item for sublist in predicted_groups for item in sublist] #if len(sublist) == 1

            gt_groups = find_components(gt_graphs[key])
            flat_gt = [item for sublist in gt_groups for item in sublist] 

            for var in all_vars:
                if var not in flat_gt:
                    gt_groups.append([var])
            
            precision, recall, f1 = pairwise_precision_recall_f1_singletons(gt_groups, predicted_groups)
            # ----------
            

            spent_time = end_time - start_time
            out_stats_row = [
                f"{key}",
                f"{max_ranks_for_ideal_recall_all}",
                f"{spent_time}",
                f"{precision}",
                f"{recall}",
                f"{f1}"
            ]

            with open(OUTPUT_STATS_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(out_stats_row)

            count_files += 1
            print(f"Finished {count_files}")


