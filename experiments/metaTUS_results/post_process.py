import pandas as pd
import os
import json



if __name__ == "__main__":
    COL_TYPES = ["LoStBench_qn", "LoStBench_lbl"]
    EMBEDDING_CATEGORIES = ["topic-embedding", "topic-filtering"]
    RESULT_TYPE = ["f1","youden"]

    QUERY_FILES_PATH = "../../LoStBench/query/"
    query_files = os.listdir(QUERY_FILES_PATH)
    query_files = [file for file in query_files if file.endswith(".csv")]

    for col_type in COL_TYPES:
        for emb_cat in EMBEDDING_CATEGORIES:
            for res_type in RESULT_TYPE:
                df = pd.read_csv(os.path.join(
                        col_type,
                        emb_cat,
                        f"meta_union_search_with_groundtruth_sorted_{res_type}.csv"
                    )
                )
            
                # group while preserving original row order
                grouped = (
                    df.groupby("query_table", sort=False)["data_lake_table"]
                    .apply(list)
                    .to_dict()
                )

                result = {
                    f"{k}.csv": [f"{k}.csv"] + [f"{v}.csv" for v in vals]
                    for k, vals in grouped.items()
                }

                #force add missing files from "query" for whom metaTUS couldn't find ranked list
                for file in query_files:
                    if file not in result:
                        result[file] = [file]

                output_filename = f"metaTus-{col_type}-{emb_cat}-{res_type}.json"
                
                with open(output_filename, 'w', encoding='utf-8') as f_out:
                    json.dump(result, f_out, indent=2, ensure_ascii=False)
                print(f"Saved Predicted ranked list for {len(result)} query items to {output_filename} successfully!!")
