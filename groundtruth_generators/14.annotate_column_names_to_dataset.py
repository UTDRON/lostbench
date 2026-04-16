'''
Default dataset column names are codes like VXXXXXX
This script renames the column names to either ANES labels or ANES Questions using the lookups we created 
'''

import pandas as pd
import json
import os


if __name__ == "__main__":

    # QUESTION_LOOKUP = "../groundtruth/questions_lookup.json"
    LABEL_LOOKUP = "../groundtruth/labels_lookup.json"
    INPUT_PATH = "../LoStBench/"
    # OUTPUT_PATH = "../LoStBench_qn/"
    OUTPUT_PATH2 = "../LoStBench_lbl/"

    with open(LABEL_LOOKUP, "r", encoding = "utf-8") as f:
        lookup = json.load(f)

    
    _folders = ["datalake", "query", "schema_matching"]

    count = 0

    for category in _folders:
        for file in os.listdir(os.path.join(INPUT_PATH, category)): #only csv files
            if not file.endswith(".csv"):
                continue

            _input_file_path = os.path.join(INPUT_PATH, category, file)
            _output_file_path = os.path.join(OUTPUT_PATH2, category, file)

            df = pd.read_csv(_input_file_path)

            new_columns = {}
            for col in df.columns:
                if col in lookup:
                    new_columns[col] = lookup[col]
                else:
                    new_columns[col] = col
            
            df = df.rename(columns=new_columns)
            df.to_csv(_output_file_path, index=False)

            count+=1
            print(f"processed{count}")