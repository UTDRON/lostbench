import os
import pandas as pd
import json

def process_folder(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    for file in files:
        # if file != "d3l_output_all_index.json":
        path = os.path.join(folder, file)

        with open(path, "r", encoding= "utf-8") as f:
            source = json.load(f)
        
        for key, value in source.items():
            source[key] = sorted(
                value,
                key=lambda t: sum(t[1]) / len(t[1]),
                reverse=True
            )
        
        with open(path, 'w', encoding = 'utf-8') as f:
            json.dump(source, f, indent=2, ensure_ascii=False)


def export_ranked_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]

    for file in files:
        if file.startswith('d3l_output'):
            path = os.path.join(folder, file)

            with open(path, "r", encoding= "utf-8") as f:
                source = json.load(f)
            
            output = {}
            for key, value in source.items():
                output[key] = []
                for item in value:
                    output[key].append(f"{item[0]}.csv")
            
            with open(os.path.join(folder,f"ranked_list_{file}"), 'w', encoding = 'utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    PROSE_QN = "prose_qn/"
    VAR_LBL = "var_labels/"

    # --- for sorting the list based on similarity values---
    process_folder(PROSE_QN)
    process_folder(VAR_LBL)

    # --- for generating ranked list for each query(following the standard
    # that are followed in others)---
    export_ranked_list(PROSE_QN)
    export_ranked_list(VAR_LBL)

