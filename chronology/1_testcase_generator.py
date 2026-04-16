import pandas as pd
import json
import ast

if __name__ == "__main__":
    GT_FILE = "../groundtruth/groundtruth_tus_true_only.csv"
    TEMP_CSV_FILE = "temp.csv"
    TEMP_JSON_FILE = "temp.json"
    GT_SM_FILE = "../groundtruth/groundtruth_sm_attribute_pairs.csv"
    TEST_CASE_FILE = "test_case.json"
    TEST_CASE_COMBO_LOOKUP_FILE = "test_case_combo_lookup.json"
    EASY_LOOKUP_FILE = "easy_lookup.json"
    UNDIRECTED_GRAPH = "undirected_graph.json"

    gt_df = pd.read_csv(GT_FILE)
    gt_sm = pd.read_csv(GT_SM_FILE)
    gt_sm["match_pairs"] = gt_sm["match_pairs"].apply(ast.literal_eval)

    '''
    remove target csvs from years earlier than source
    '''
    temp_df = gt_sm[
        gt_sm['schema1'].str[5:9].astype(int) < gt_sm['schema2'].str[5:9].astype(int)
    ]
    # temp_df.to_csv(TEMP_CSV_FILE, index=False)
    '''
    -----------------------------------------
    '''

    '''
    sort alphabetically based on source file names and year
    '''
    sorted_df = temp_df.sort_values(by="schema1").reset_index(drop=True)
    # sorted_df.to_csv(TEMP_CSV_FILE, index=False)
    '''
    -----------------------------------------
    '''

    '''
    sort target file names for each source file group alphabetically
    '''
    sorted_df = sorted_df.sort_values(by=["schema1", "schema2"]).reset_index(drop=True)
    # sorted_df = sorted_df[sorted_df["num_matches"] > 1].reset_index(drop=True) # just for checking; can remove later
    # sorted_df.to_csv(TEMP_CSV_FILE, index=False)
    '''
    -----------------------------------------
    '''

    '''
    for choosing contiguous and chronological schemas for test cases
    '''
    sorted_dict = sorted_df.groupby("schema1")["schema2"].apply(list).to_dict()

    '''
    possible_combo:
    '''
    filtered_dict = {}
    leading_schemas_in_testcase = set() 
    test_case = {}
    
    for qtable, dl_list in sorted_dict.items():
        possible_combo = []
        for item in dl_list:
            possible_combo.append((qtable, item))
    
        for i in range(len(dl_list)):
            for j in range(i+1, len(dl_list)):
                possible_combo.append((dl_list[i], dl_list[j]))
        
        result = all(
            ((sorted_df['schema1'] == a) & (sorted_df['schema2'] == b)).any()
            for (a, b) in possible_combo
        )

        if result and len(possible_combo) > 1:
            if qtable not in leading_schemas_in_testcase:
                filtered_dict[qtable] = []
                for (a,b) in possible_combo:
                    match_pairs = sorted_df[(sorted_df['schema1'] == a) & (sorted_df['schema2'] == b)]['match_pairs'].iloc[0]
                    filtered_dict[qtable].append(
                        {
                            'schema1': a,
                            'schema2': b,
                            'match_pairs': match_pairs
                        }
                    )
                
                leading_schemas_in_testcase.update(dl_list)
                test_case[qtable] = dl_list

    with open(TEST_CASE_COMBO_LOOKUP_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(filtered_dict, f_out, indent=2, ensure_ascii=False)
    print(f"Saved {len(filtered_dict)} cases of lookups for combinations of test cases to {TEST_CASE_COMBO_LOOKUP_FILE} successfully")
            
    '''
    -----------------------------------------
    '''
    with open(TEST_CASE_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(test_case, f_out, indent=2, ensure_ascii=False)
    print(f"Saved {len(test_case)} test cases {TEST_CASE_FILE} successfully")

    '''
    -----------------------------------------
    '''

    easy_lookup = {}
    for file, items in filtered_dict.items():
        easy_lookup[file] = []
        for item in items:
            easy_lookup[file].extend(list(item['match_pairs']))
        
    
    with open(EASY_LOOKUP_FILE, 'w', encoding='utf-8') as f_out:
        json.dump(easy_lookup, f_out, indent=2, ensure_ascii=False)
    print(f"Saved {len(easy_lookup)} test cases in {EASY_LOOKUP_FILE} successfully")

    '''
    -----------------------------------------
    '''

    for file,items in easy_lookup.items():
        to_append = []
        for [a,b] in easy_lookup[file]:
            if [b,a] in easy_lookup[file]:
                print(f"[{b},{a}] already in list")
            else:
                to_append.append([b,a])
        easy_lookup[file].extend(to_append)
    
    # with open(UNDIRECTED_GRAPH, 'w', encoding='utf-8') as f_out:
    #     json.dump(easy_lookup, f_out, indent=2, ensure_ascii=False)
    # print(f"Saved {len(easy_lookup)} test cases in {EASY_LOOKUP_FILE} successfully")
    '''
    -----------------------------------------
    '''

    undirected_graph = {}
    for file,items in easy_lookup.items():
        undirected_graph[file] = {}
        for [a,b] in items:
            if a not in undirected_graph[file]:
                undirected_graph[file][a] = []
            undirected_graph[file][a].append(b)
    
    with open(UNDIRECTED_GRAPH, 'w', encoding='utf-8') as f_out:
        json.dump(undirected_graph, f_out, indent=2, ensure_ascii=False)
    print(f"Saved {len(undirected_graph)} test cases in {UNDIRECTED_GRAPH} successfully")
    '''
    -----------------------------------------
    '''
    