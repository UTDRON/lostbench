import os
import sys
import json
import time
import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma
from valentine.algorithms import Cupid
from valentine.algorithms import DistributionBased
from valentine.algorithms import JaccardDistanceMatcher
from valentine.algorithms import SimilarityFlooding
import ast
import re
import pdb

PATH_TO_DATA_FOLDERS = "../LoStBench_qn/schema_matching/"
PATH_TO_GT_CSV = "../groundtruth/groundtruth_sm_test_case.csv"
PATH_TO_LOOKUP = "../groundtruth/questions_lookup.json"
OUTPUT_DIR = "../experiments/valentine/"
RANKED_LIST_DIR = "../experiments/valentine/ranked_lists/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RANKED_LIST_DIR, exist_ok=True)


def robust_reverse_lookup(text, reverse_lookup):
    """Look up text in reverse_lookup, stripping pandas duplicate column suffixes if needed.
    
    When pandas reads CSV files with duplicate column names, it appends .1, .2, etc.
    to make them unique. This function stripped those suffixes before lookup which gave us the full question instead of codes.
    """
    result = reverse_lookup.get(text)
    if result is not None:
        return result
    # Stripping pandas-added .N suffix for duplicate columns
    stripped = re.sub(r'\.\d+$', '', text)
    if stripped != text:
        result = reverse_lookup.get(stripped)
        if result is not None:
            return result
    return text  # Return original text if no match found


MAX_CUPID_TOKENS = 15  # Cap column names for Cupid to avoid O(T²) token-pair explosion

def truncate_column_name(name, max_tokens=MAX_CUPID_TOKENS):
    """
    Truncate a column name to at most max_tokens words.
    
    Cupid tokenizes column names and compares every token pair via WordNet.
    Long question texts (50-800 tokens) cause extreme slowdowns. Keeping the
    first ~15 words preserves enough semantic content for meaningful matching.
    
    Preserves pandas-added .N suffixes for duplicate columns (e.g. "text.1").
    """

    # Strip pandas .N suffix, truncate, then re-attach
    suffix = ''
    match = re.search(r'\.\d+$', name)
    if match:
        suffix = match.group()
        name = name[:match.start()]
    words = name.split()
    if len(words) <= max_tokens:
        return name + suffix
    return ' '.join(words[:max_tokens]) + suffix


def find_file_path(filename, folder):
    """
    Searches for a filename in the list of provided folders.
    Returns the full path if found, else None.
    """
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path):
        # print(full_path)
        return full_path
    return None

def ranked_list(rank_matches, reverse_lookup=None):
    ranked_list_dict = {}
    bipartite_matches = {}
    for key in rank_matches:
        first_id = key[0][1]
        second_id = key[1][1]
        # Convert question text to variable codes if reverse_lookup is provided
        if reverse_lookup is not None:
            first_id = robust_reverse_lookup(first_id, reverse_lookup)
            second_id = robust_reverse_lookup(second_id, reverse_lookup)
        bipartite_matches[(first_id, second_id)] = rank_matches[key]
        if first_id not in ranked_list_dict:
            ranked_list_dict[first_id] = [(first_id, second_id)]
        else:
            ranked_list_dict[first_id].append((first_id, second_id))
    return ranked_list_dict, bipartite_matches

def mrr_get_matchings(rank_matches):
    ranked_dict = {}
    for key, similarity in rank_matches.items():
        first_id = key[0][1]
        second_id = key[1][1]
        pair = (first_id, second_id)
        if first_id not in ranked_dict:
            ranked_dict[first_id] = {pair: similarity}
        else:
            ranked_dict[first_id][pair] = similarity
    # print(ranked_dict)
    return ranked_dict

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

def extract_metrics(metrics_dict):
    """
    Extract all metrics from the Valentine metrics dictionary.
    Returns a dict with all metric values.
    """
    return {
        'All_Recall': metrics_dict.get('Recall', None),
        'All_Precision': metrics_dict.get('Precision', None),
        'All_F1Score': metrics_dict.get('F1Score', None),
        'All_PrecisionTop10Percent': metrics_dict.get('PrecisionTop10Percent', None),
        'All_RecallAtSizeofGroundTruth': metrics_dict.get('RecallAtSizeofGroundTruth', None),
    }

def get_ranked_matches(matches, reverse_lookup=None):
    matchings, bipartite_matches = ranked_list(matches, reverse_lookup=reverse_lookup)
    return matchings, bipartite_matches

def run_matcher_and_record(matcher, matcher_name, df_src, df_tgt, ground_truth, 
                           file_src, file_tgt, ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches):
    """
    Run a matcher and return a result dictionary with all required columns,
    plus the raw matches object for reuse (e.g., ranked list generation).
    """
    result = {
        'benchmark': 'ANES',
        'dataset': 'anes_qn',
        'source_table': file_src,
        'target_table': file_tgt,
        'ncols_src': ncols_src,
        'ncols_tgt': ncols_tgt,
        'nrows_src': nrows_src,
        'nrows_tgt': nrows_tgt,
        'nmatches': nmatches,
        'method': matcher_name,
        'runtime': None,
        'mrr': None,
        'All_Recall': None,
        'All_Precision': None,
        'All_F1Score': None,
        'All_PrecisionTop10Percent': None,
        'All_RecallAtSizeofGroundTruth': None,
        'One2One_RecallAtSizeofGroundTruth': None
    }
    matches = None
    
    try:
        start_time = time.time()
        matches = valentine_match(df_src, df_tgt, matcher)
        matchings = mrr_get_matchings(matches)
        mrr_value = mrr(ground_truth, matchings)
        result['mrr'] = mrr_value
        metrics = matches.get_metrics(ground_truth)
        print(metrics)
        runtime = time.time() - start_time
        
        result['runtime'] = runtime
        metric_values = extract_metrics(metrics)
        result.update(metric_values)
    
        # print(f"{matcher_name}: {metrics} | Runtime: {runtime:.4f}s")
    except Exception as e:
        print(f"{matcher_name}: Error - {e}")
    
    return result, matches

MATCHERS = {
    'JaccardDistanceMatcher': JaccardDistanceMatcher(),  # ~0.004s
    # 'JaccardDistanceMatcher_t0.7': JaccardDistanceMatcher(threshold_dist=0.7),
    # 'DistributionBased': DistributionBased(),  # ~0.8s
    # 'SimilarityFlooding_basic_invavg': SimilarityFlooding(coeff_policy='inverse_average', formula='basic'),
    # 'SimilarityFlooding_basic_invprod': SimilarityFlooding(coeff_policy='inverse_product', formula='basic'),
    # 'SimilarityFlooding_formulaA_invavg': SimilarityFlooding(coeff_policy='inverse_average', formula='formula_a'),
    # 'SimilarityFlooding_formulaA_invprod': SimilarityFlooding(coeff_policy='inverse_product', formula='formula_a'),
    # 'SimilarityFlooding_formulaB_invavg': SimilarityFlooding(coeff_policy='inverse_average', formula='formula_b'),
    # 'SimilarityFlooding_formulaB_invprod': SimilarityFlooding(coeff_policy='inverse_product', formula='formula_b'),
    # 'Coma': Coma(use_instances=False),  # moderate
    # 'Coma++': Coma(use_instances=True),  # slower
    # 'Cupid': Cupid(),  # SLOWEST (minutes)
}

# CSV columns
CSV_COLUMNS = [
    'benchmark', 'dataset', 'source_table', 'target_table',
    'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
    'method', 'runtime', 'mrr', 'All_Recall',
    'All_Precision', 'All_F1Score',
    'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth', 'One2One_RecallAtSizeofGroundTruth'
]

CSV_COLUMNS_RANKED_LIST = [
    'schema1', 'schema2',
    'match_pairs',
    'ranked_list'
]

gt_df = pd.read_csv(PATH_TO_GT_CSV)

# Load question lookup and create reverse lookup (question text -> variable code)
with open(PATH_TO_LOOKUP, 'r') as f:
    question_lookup = json.load(f)

# Create reverse lookup: question text -> variable code
reverse_lookup = {v.replace('\n', ' '): k for k, v in question_lookup.items()}

# Create truncated reverse lookup: truncated question text -> variable code
truncated_reverse_lookup = {truncate_column_name(v.replace('\n', ' ')): k for k, v in question_lookup.items()}

# Create CSV files with headers for each matcher
for name in MATCHERS.keys():
    output_path = os.path.join(OUTPUT_DIR, f"results_{name.replace(' ', '_')}.csv")
    ranked_output_path = os.path.join(RANKED_LIST_DIR, f"valentine_ranked_list_{name.replace(' ', '_')}.csv")
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(output_path, index=False)
    pd.DataFrame(columns=CSV_COLUMNS_RANKED_LIST).to_csv(ranked_output_path, index=False)

# Process one matcher at a time (reduces overhead for expensive matchers like Coma, Cupid)
for matcher_name, matcher in MATCHERS.items():
    print(f"\n{'='*60}")
    print(f"Starting matcher: {matcher_name}")
    print(f"{'='*60}")
    
    output_path = os.path.join(OUTPUT_DIR, f"results_{matcher_name.replace(' ', '_')}.csv")
    ranked_output_path = os.path.join(RANKED_LIST_DIR, f"valentine_ranked_list_{matcher_name.replace(' ', '_')}.csv")
    
    # Iterate through all the rows in the Ground Truth CSV for this matcher
    for index, row in gt_df.iterrows():
        file_src = row["schema1"]
        file_tgt = row["schema2"]
        nmatches = row["num_matches"]

        ground_truth_raw = ast.literal_eval(row["match_pairs"])
        
        # Keep variable codes directly (no conversion to question text)
        # Valentine expects ground truth as simple tuples: (col1, col2)
        ground_truth = [(pair[0], pair[1]) for pair in ground_truth_raw]

        # mrr = mrr(ground_truth, gt_df) # Here the ground_truht is the code from thr groundtruth match dictionary then list, 
        # print(f"\n--- Processing: {file_src} <-> {file_tgt} ---")
        # print(f"Ground truth pairs: {ground_truth}")
        
        path_src = find_file_path(file_src, PATH_TO_DATA_FOLDERS)
        path_tgt = find_file_path(file_tgt, PATH_TO_DATA_FOLDERS)
        
        if not path_src or not path_tgt:
            print(f"SKIP: Could not find one of the files: {file_src} or {file_tgt}")
            continue
        
        try:
            df_src = pd.read_csv(path_src, low_memory=False)
            df_tgt = pd.read_csv(path_tgt, low_memory=False)
            
            # Sanitize column names by replacing newlines with spaces
            df_src.columns = [col.replace('\n', ' ') for col in df_src.columns]
            df_tgt.columns = [col.replace('\n', ' ') for col in df_tgt.columns]
            
            # Ground truth uses variable codes - convert to question text for non-DistributionBased matchers
            ground_truth_text = [(question_lookup.get(pair[0], pair[0]).replace('\n', ' '), 
                                  question_lookup.get(pair[1], pair[1]).replace('\n', ' ')) for pair in ground_truth]
            
            ncols_src = len(df_src.columns)
            ncols_tgt = len(df_tgt.columns)
            nrows_src = len(df_src)
            nrows_tgt = len(df_tgt)
            
            # Create DataFrames with variable codes for DistributionBased (short column names)
            df_src_codes = df_src.copy()
            df_tgt_codes = df_tgt.copy()
            df_src_codes.columns = [robust_reverse_lookup(col, reverse_lookup) for col in df_src.columns]
            df_tgt_codes.columns = [robust_reverse_lookup(col, reverse_lookup) for col in df_tgt.columns]

            # Create DataFrames with truncated question text for Cupid
            # Cupid needs semantic text (not codes) but is O(T1*T2) per column pair
            df_src_truncated = df_src.copy()
            df_tgt_truncated = df_tgt.copy()
            df_src_truncated.columns = [truncate_column_name(col) for col in df_src.columns]
            df_tgt_truncated.columns = [truncate_column_name(col) for col in df_tgt.columns]

            ground_truth_truncated = [(truncate_column_name(pair[0]), truncate_column_name(pair[1]))
                                      for pair in ground_truth_text]

        except Exception as e:
            print(f"Error reading CSVs: {e}")
            continue


        # For DistributionBased, use variable codes (short names) to avoid file name too long error
        # For Cupid, use truncated question text to avoid O(T²) token-pair explosion while preserving semantic content for WordNet matching
        if matcher_name in ('DistributionBased'): # Use this to use the lookup for only distributionbased
            df_src_use = df_src_codes
            df_tgt_use = df_tgt_codes
            gt_use = ground_truth  # variable codes
        if matcher_name in ('Cupid'):
            df_src_use = df_src_truncated
            df_tgt_use = df_tgt_truncated
            gt_use = ground_truth_truncated  # truncated question text
        else:
            df_src_use = df_src
            # df_src_use = df_src_codes
            # df_tgt_use = df_tgt_codes
            df_tgt_use = df_tgt
            gt_use = ground_truth_text  # question text
            # gt_use = ground_truth
        
        result, matches = run_matcher_and_record(
            matcher=matcher,
            matcher_name=matcher_name,
            df_src=df_src_use,
            df_tgt=df_tgt_use,
            ground_truth=gt_use,
            file_src=file_src,
            file_tgt=file_tgt,
            ncols_src=ncols_src,
            ncols_tgt=ncols_tgt,
            nrows_src=nrows_src,
            nrows_tgt=nrows_tgt,
            nmatches=nmatches
        )

        # Reuse the matches from run_matcher_and_record
        if matches is not None:
            # For Cupid, matches contain truncated text; use truncated_reverse_lookup
            rl = truncated_reverse_lookup if matcher_name in ('Cupid') else reverse_lookup
            ranked_matches, bipartite_matches = get_ranked_matches(matches, reverse_lookup=rl)
        else:
            ranked_matches, bipartite_matches = {}, {}
        
        one2one_matches = greedyBipartiteMatching(bipartite_matches)

        df_ranked_result = pd.DataFrame([[file_src, file_tgt, one2one_matches, ranked_matches]], columns=CSV_COLUMNS_RANKED_LIST)
        df_ranked_result.to_csv(ranked_output_path, mode='a', header=False, index=False)

        result['One2One_RecallAtSizeofGroundTruth'] = one2one_recall_at_gt(ground_truth_raw, one2one_matches)
        df_result = pd.DataFrame([result], columns=CSV_COLUMNS)
        df_result.to_csv(output_path, mode='a', header=False, index=False)
    
    print(f"Completed matcher: {matcher_name}")
