'''
For Creating a separate directory for only those files that are in the test case pairs for schema matching
'''

import os
import pandas as pd
import shutil

def main():
    gt_df = pd.read_csv(GT_FILE)

    for _, row in gt_df.iterrows():
        src_file = os.path.join(SRC_FOLDER, row["schema1"])
        shutil.copy(src_file, DEST_FOLDER)

        tgt_file = os.path.join(SRC_FOLDER, row["schema2"])
        shutil.copy(tgt_file, DEST_FOLDER)
    
    files = os.listdir(DEST_FOLDER)
    files = [file for file in files if file != '.DS_Store']
    print(f"Copied {len(files)} files successfully to {DEST_FOLDER}")


if __name__ == "__main__":
    SRC_FOLDER = "../LoStBench/datalake/"
    DEST_FOLDER = "../LoStBench/schema_matching/"
    GT_FILE = "../groundtruth/groundtruth_sm_test_case.csv"

    main()