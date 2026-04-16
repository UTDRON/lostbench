import ast
import numpy as np
import pandas as pd


def get_max_rank_for_ideal_recall(gt_pairs, ranked_list):
    for  i, (u,v) in enumerate(ranked_list):
        if [u,v] in gt_pairs:
            return i+1
    return 50


def exists_as_first(key, pairs):
    return any(u == key for u, _ in pairs)


if __name__ == "__main__":

    GT_FILE = "../groundtruth/groundtruth_sm_test_case.csv"

    # MODEL_TO_PROCESS = "../experiments/DDM/sm_results/ddm_sm_lostbench_qn_match_output.csv"
    # METRIC_FILE_TO_PROCESS = "../experiments/DDM/sm_results/ddm_sm_lostbench_qn_copy.csv"

    MODELS_TO_PROCESS = [
                        "../experiments/DDM/sm_results/ddm_sm_lostbench_qn_match_output.csv", #Done
                        "../experiments/DDM/sm_results/ddm_sm_lostbench_lbl_match_output.csv", #Done
                        "../experiments/DDM/sm_results/ddm_llm_sm_lostbench_qn_match_output.csv", #Done
                        "../experiments/magneto/output_pairs/magneto_sm_lostbench_qn_match_output.csv", #Done
                        "../experiments/magneto/output_pairs/magnetogpt_sm_lostbench_qn_match_output.csv", #Done

                        "../experiments/valentine/ranked_lists/valentine_ranked_list_Coma.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_Coma++.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_DistributionBased.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_JaccardDistanceMatcher_t0.7.csv",
                       
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_basic_invavg.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_basic_invprod.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_formulaA_invavg.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_formulaA_invprod.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_formulaB_invavg.csv",
                        "../experiments/valentine/ranked_lists/valentine_ranked_list_SimilarityFlooding_formulaB_invprod.csv",
                         "../experiments/valentine/ranked_lists/valentine_ranked_list_JaccardDistanceMatcher.csv",

    ]

    METRICS_FILES_TO_PROCESS = [
                        "../experiments/DDM/sm_results/ddm_sm_lostbench_qn.csv", #Done
                        "../experiments/DDM/sm_results/ddm_sm_lostbench_lbl.csv", #Done
                        "../experiments/DDM/sm_results/ddm_llm_sm_lostbench_qn.csv", #Done
                        "../experiments/magneto/output_metrics/magneto_sm_lostbench_qn_metrics.csv", #Done
                        "../experiments/magneto/output_metrics/magnetogpt_sm_lostbench_qn_metrics.csv", #Done
                        
                        "../experiments/valentine/results_Coma.csv",
                        "../experiments/valentine/results_Coma++.csv",
                        "../experiments/valentine/results_DistributionBased.csv",
                        "../experiments/valentine/results_JaccardDistanceMatcher_t0.7.csv",
                        "../experiments/valentine/results_SimilarityFlooding_basic_invavg.csv",
                        "../experiments/valentine/results_SimilarityFlooding_basic_invprod.csv",
                        "../experiments/valentine/results_SimilarityFlooding_formulaA_invavg.csv",
                        "../experiments/valentine/results_SimilarityFlooding_formulaA_invprod.csv",
                        "../experiments/valentine/results_SimilarityFlooding_formulaB_invavg.csv",
                        "../experiments/valentine/results_SimilarityFlooding_formulaB_invprod.csv",

                         "../experiments/valentine/results_JaccardDistanceMatcher.csv",

    ]

    gt_df = pd.read_csv(GT_FILE)
    gt_df["match_pairs"] = gt_df["match_pairs"].apply(ast.literal_eval)


    for i in range(len(MODELS_TO_PROCESS)):
        MODEL_TO_PROCESS = MODELS_TO_PROCESS[i]
        MODEL_METRICS_TO_PROCESS = METRICS_FILES_TO_PROCESS[i]

        model_ranked_lists_df = pd.read_csv(MODEL_TO_PROCESS)
        model_ranked_lists_df["ranked_list"] = model_ranked_lists_df["ranked_list"].apply(ast.literal_eval)
        model_metrics_df = pd.read_csv(MODEL_METRICS_TO_PROCESS)

        max_ranks_col = []

        for i in range(len(model_ranked_lists_df)):
            gt_pairs = gt_df['match_pairs'][i]
            ranked_list_dict = model_ranked_lists_df['ranked_list'][i]

            max_ranks = []

            if ranked_list_dict:
                for key, ranked_list in ranked_list_dict.items():
                    # if i == 0:
                    #     print(key, ranked_list)

                    if exists_as_first(key, gt_pairs):
                        max_rank_for_ideal_recall = get_max_rank_for_ideal_recall(gt_pairs, ranked_list)

                        if max_rank_for_ideal_recall == 50:
                            max_rank_for_ideal_recall = int(model_metrics_df['ncols_tgt'][i])
                        
                        max_ranks.append(max_rank_for_ideal_recall)
            else:
                max_ranks.extend([int(model_metrics_df['ncols_tgt'][i])] * int(model_metrics_df['nmatches'][i]))

            
            max_ranks_col.append(max_ranks)
            # break
        
        model_metrics_df["max_rank_for_ideal_recall"] = max_ranks_col
        model_metrics_df["avg_max_rank_for_ideal_recall"] = model_metrics_df["max_rank_for_ideal_recall"].apply(
            lambda lst: np.mean(lst) if lst else np.nan
        )

        print("Average of avg_max_rank_for_ideal_recall:", model_metrics_df["avg_max_rank_for_ideal_recall"].mean())

        model_metrics_df.to_csv(MODEL_METRICS_TO_PROCESS, index=False)
        print(f"Saved Updated metrics file to {MODEL_METRICS_TO_PROCESS} successfully!!")
