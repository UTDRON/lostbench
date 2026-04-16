'''
For the Generation of a Lookup dictionary for
Table name -> List of Column Names (ANES VAR CODE NAMES)
'''
import json
import pandas as pd
import os


def main():
    tables = os.listdir(TABLES_PATH)
    tables.remove('.DS_Store')

    num_done = 0
    cols_lookup = {}

    for table in tables:
        try:
            table_path = f"{TABLES_PATH}{table}"
            df = pd.read_csv(table_path)
            cols = list(df.columns)
            cols_lookup[table] = cols 
            num_done +=1
            print(f"{num_done} done")
        
        except Exception as e:
            print("Error: ", e)
            print("Error type:", type(e).__name__)
    
    return cols_lookup
    

if __name__ == "__main__":
    '''
    Change these paths based on your codebase directory
    '''
    TABLES_PATH = f"../../../release_data/data_instances/type1_null_filled/tables/"
    OUTPUT_PATH = "../groundtruth/table_columns_lookup.json"

    cols_lookup = main()

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(cols_lookup, f_out, indent=2, ensure_ascii=False)
    print(f"Saved Table->Columns_List lookup to {OUTPUT_PATH} successfully")
