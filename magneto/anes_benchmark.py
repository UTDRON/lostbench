"""
1. Paste this file inside the root/experiments/benchmarks/ folder of magneto project directory structure
2. keep the LoStBench dataset folder inside root/data/. Keep datalake folder inside LoStBench/ANES/.
3. You'd also need question_lookup.json  and schema_matching_attribute_pairs_groundtruth.csv inside the LoStBench/ANES/ folder.
4. Before running this script you'll need to run a preprocessing script- lostbench_preprocessor_for_magneto.py
"""

import pprint
import os
import sys
import json
import pandas as pd
import time
import datetime
import argparse
from valentine import valentine_match
from valentine.algorithms import Coma
import valentine.algorithms.matcher_results as matcher_results
import ast
import csv
# from algorithms.magneto.magneto.utils.utils import get_codes_and_questions, get_codes_from_qn
import warnings

warnings.simplefilter("ignore", FutureWarning)


project_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_path))


from experiments.benchmarks.benchmark_utils import (
    compute_mean_ranking_reciprocal,
    create_result_file,
    record_result,
)
from algorithms.magneto.magneto.magneto import Magneto


pp = pprint.PrettyPrinter(indent=4)


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [
        (match["source_column"], match["target_column"]) for match in data["matches"]
    ]
    return matchings


def get_matcher(method, model_name=None, mode="header_values_default"):
    if method == "Coma":
        return Coma()
    elif method == "ComaInst":
        return Coma(use_instances=True, java_xmx="10096m")
    elif method == "Magneto":
        return Magneto()
    elif method == "MagnetoFT":
        model_path = os.path.join(
            project_path,
            "models",
            model_name,
        )
        return Magneto(encoding_mode=mode, embedding_model=model_path)
    elif method == "MagnetoGPT":
        return Magneto(use_bp_reranker=False, use_gpt_reranker=True)
    elif method == "MagnetoFTGPT":
        model_path = os.path.join(
            project_path,
            "models",
            model_name,
        )
        return Magneto(
            encoding_mode=mode,
            embedding_model=model_path,
            use_bp_reranker=False,
            use_gpt_reranker=True,
        )

def get_codes_and_questions(src_filename, tgt_filename):
    
    src_codes = pd.read_csv(os.path.join("../../data/LoStBench/ANES/datalake/", src_filename), nrows=0).columns.tolist()
    tgt_codes = pd.read_csv(os.path.join("../../data/LoStBench/ANES/datalake/", tgt_filename), nrows=0).columns.tolist()

    src_qns = pd.read_csv(os.path.join("../../data/LoStBench/ANES/datalake_qn/", src_filename), nrows=0).columns.tolist()
    tgt_qns = pd.read_csv(os.path.join("../../data/LoStBench/ANES/datalake_qn/", tgt_filename), nrows=0).columns.tolist()

    return src_codes, tgt_codes, src_qns, tgt_qns

def get_codes_from_qn(query, codes, questions):
    idx = questions.index(query)
    return codes[idx]

    
def post_process(source_file, target_file, method, matches, matches_121, OUTPUT_PAIRS_PATH):
    #--- some wierd difference in filename pattern seemn received 
    anes_idx_src = source_file.find("ANES_")
    src_filename = source_file[anes_idx_src:]
    anes_idx_tgt = target_file.find("ANES_")
    tgt_filename = target_file[anes_idx_tgt:]

    src_codes, tgt_codes, src_qns, tgt_qns = get_codes_and_questions(src_filename, tgt_filename)

    #--- process 121 matches ---
    ranked_matches_121 = [
        (pair[0][1], pair[1][1])   # (source_qn, target_qn)
        for pair, score in sorted(
            matches_121.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
    coded_matches_121 = [[get_codes_from_qn(u,src_codes, src_qns),get_codes_from_qn(v, tgt_codes, tgt_qns)] for (u,v) in ranked_matches_121]

    #--- process all matches ---
    ranked_matches_list = [
        (pair[0][1], pair[1][1])   # (source_qn, target_qn)
        for pair, score in sorted(
            matches.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
    coded_matches = [[get_codes_from_qn(u,src_codes, src_qns),get_codes_from_qn(v, tgt_codes, tgt_qns)] for (u,v) in ranked_matches_list]
    matches_dict = {}
    for [u,v] in coded_matches:
        if u in matches_dict:
            matches_dict[u].append((u,v))
        else:
            matches_dict[u] = [(u,v)]


    #---new row to append to output csv
    out_row = [
        f"{src_filename}",
        f"{tgt_filename}",
        f"{method}",
        f"{coded_matches_121}",
        f"{matches_dict}"
    ]
    
    # with open(OUTPUT_PAIRS_PATH, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(out_row)
    

def run_valentine_benchmark_on_lostbench(
    BENCHMARK="LoStBench",
    DATASET="ANES",
    ROOT="../../data/LoStBench/ANES/datalake/",
    MODE="header_values_default",
):

    HEADER = [
        "benchmark",
        "dataset",
        "source_table",
        "target_table",
        "ncols_src",
        "ncols_tgt",
        "nrows_src",
        "nrows_tgt",
        "nmatches",
        "method",
        "runtime",
        "mrr",
        "All_Precision",
        "All_F1Score",
        "All_Recall",
        "All_PrecisionTop10Percent",
        "All_RecallAtSizeofGroundTruth",
        "One2One_Precision",
        "One2One_F1Score",
        "One2One_Recall",
        "One2One_PrecisionTop10Percent",
        "One2One_RecallAtSizeofGroundTruth",
    ]

    dataset_name = DATASET.lower()
    model_names = [
        f"mpnet-{dataset_name}-{MODE}-exact_semantic-16-0.5.pth",
        # f"mpnet-{dataset_name}-{MODE}-exact-16-0.5.pth",
        # f"mpnet-{dataset_name}-{MODE}-semantic-16-0.5.pth",
    ]

    ## --- For saving the output match pairs ---
    OUTPUT_PAIRS_PATH = "../../../anes_gss/experiments/magneto/output_pairs/magneto_sm_lostbench_qn_match_output_test.csv"
    columns_output_pairs = [
        "schema1",
        "schema2",
        "method",
        "match_pairs",
        "ranked_list"
    ]
    out_matches_df = pd.DataFrame(columns=columns_output_pairs)
    # out_matches_df.to_csv(OUTPUT_PAIRS_PATH, index=False) #Save to csv- headers only
    #################################

    for model_name in model_names:

        print(f"Model: {model_name}")

        results_dir = os.path.join(
            project_path, "results", "benchmarks", BENCHMARK, DATASET
        )
        result_file = os.path.join(
            results_dir,
            DATASET
            # + "-"
            # + model_name.replace("-16-0.5.pth", "").replace("mpnet-", "")
            + "_results" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv",
        )
        print(result_file)

        create_result_file(results_dir, result_file, HEADER)

        matcher = Magneto()

        for _, row in gt_df.iterrows():
            source_file = row['schema1']
            target_file = row['schema2']
            

        # for folder in os.listdir(ROOT):
        #     if folder == ".DS_Store":
        #         continue

            # source_file = os.path.join(ROOT, folder, folder.lower() + "_source.csv")
            # target_file = os.path.join(ROOT, folder, folder.lower() + "_target.csv")
            # mapping_file = os.path.join(ROOT, folder, folder.lower() + "_mapping.json")

            source_file = os.path.join(ROOT, source_file)
            target_file = os.path.join(ROOT, target_file)

            df_source = pd.read_csv(source_file)
            df_target = pd.read_csv(target_file)
            # ground_truth = extract_matchings(open(mapping_file).read())
            ground_truth_codes = row['match_pairs']
            # print(ground_truth_codes)

            ground_truth = [(qn_lookup.get(i, i), qn_lookup.get(j, j)) for (i, j) in ground_truth_codes]
            # print(ground_truth)
            # break
            
            ncols_src = str(df_source.shape[1])
            ncols_tgt = str(df_target.shape[1])
            nrows_src = str(df_source.shape[0])
            nrows_tgt = str(df_target.shape[0])

            nmatches = len(ground_truth)

            if len(ground_truth) == 0:
                continue

            # matchers = ["Magneto", "MagnetoGPT", "MagnetoFT", "MagnetoFTGPT"]
            matchers = ["Magneto", "MagnetoGPT"]
            # matchers = ["Magneto"]
            

            for method_name in matchers:

                print("Running matcher: ", method_name)
                matcher = get_matcher(method_name, model_name, MODE)

                start_time = time.time()

                if "Coma" in method_name:
                    matches = valentine_match(df_source, df_target, matcher)
                else:
                    matches = matcher.get_matches(df_source, df_target, source_file, target_file)

                end_time = time.time()
                runtime = end_time - start_time

                mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics["RecallAtSizeofGroundTruth"]

                print(
                    method_name,
                    " with MRR Score: ",
                    mrr_score,
                    " and RecallAtGT: ",
                    recallAtGT,
                )

                matches_121 = matches.one_to_one()
                post_process(source_file, target_file,method_name, matches, matches_121,OUTPUT_PAIRS_PATH)

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)
             
                

                source_file = source_file.split("/")[-1]
                target_file = target_file.split("/")[-1]

                result = [
                    BENCHMARK,
                    DATASET,
                    source_file,
                    target_file,
                    ncols_src,
                    ncols_tgt,
                    nrows_src,
                    nrows_tgt,
                    nmatches,
                    method_name,
                    runtime,
                    mrr_score,
                    all_metrics["Precision"],
                    all_metrics["F1Score"],
                    all_metrics["Recall"],
                    all_metrics["PrecisionTop10Percent"],
                    all_metrics["RecallAtSizeofGroundTruth"],
                    one2one_metrics["Precision"],
                    one2one_metrics["F1Score"],
                    one2one_metrics["Recall"],
                    one2one_metrics["PrecisionTop10Percent"],
                    one2one_metrics["RecallAtSizeofGroundTruth"],
                ]

                record_result(result_file, result)


def main():
    parser = argparse.ArgumentParser(description="Run the LoStBench benchmark")

    parser.add_argument(
        "--benchmark",
        type=str,
        help="The benchmark to run. Default is LoStBench",
        default="lostbench",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to run the benchmark on. Default is ANES",
        default="anes",
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="serialization mode",
        default="header_values_default",
    )

    args = parser.parse_args()
    dataset_dict = {
        "anes": "ANES",
    }
    path_dict = {
        "ANES": "../../data/LoStBench/ANES/datalake_qn/",
    }
    DATASET = dataset_dict[args.dataset]
    root = path_dict[DATASET]

    run_valentine_benchmark_on_lostbench(args.benchmark, DATASET, root, args.mode)

if __name__ == "__main__":
    GT_FILE = "../../data/LoStBench/ANES/groundtruth_sm_test_case.csv"
    QN_LOOKUP_FILE = "../../data/LoStBench/ANES/questions_lookup.json"
    LBL_LOOKUP_FILE = "../../data/LoStBench/ANES/labels_lookup.json"

    gt_df = pd.read_csv(GT_FILE)
    gt_df["match_pairs"] = gt_df["match_pairs"].apply(
        lambda x: [tuple(pair) for pair in ast.literal_eval(x)]
    )

    with open(QN_LOOKUP_FILE, "r", encoding = "utf-8") as f:
        qn_lookup = json.load(f)
    
    main()
