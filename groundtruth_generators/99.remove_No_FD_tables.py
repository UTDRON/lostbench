'''
-> Actual src has 2056 files
-> SANTOS couldn't evaluate FD for 38 files, 
which were manually removed in non-release version
-> this scripts finds those 38 files and removes 
them from new release LoStBench 


NOTE: Turns out all of those 38 problematic files were removed in
the preceeding steps; All of them had columns that were completely empty




NOTE: UPDATE: 10 files in "../groundtruth/no_FDs.txt" were further needed to be removed
because santos couldn't find FDs (maybe this was missed before)
Manually remove it or with command line command: 

"cat groundtruth/no_FDs.txt | xargs -I {} rm -v LoStBench/datalake/{} LoStBench/query/{}"

'''
import pandas as pd
import os


if __name__ == "__main__":
    '''
    Change these paths based on your codebase directory
    '''
    SRC_PATH = "../../../release_data/data_instances/type1_null_filled/tables/"
    OLD_DATALAKE_PATH = "../anes/datalake/"
    LOSTBENCH_PATH = "../LoStBench/datalake/"

    src_tables = os.listdir(SRC_PATH)
    src_tables = [table for table in src_tables if table != '.DS_Store']
    src_tables = set(src_tables)
    
    old_datalake_tables = os.listdir(OLD_DATALAKE_PATH)
    old_datalake_tables = [table for table in old_datalake_tables if table != '.DS_Store']
    old_datalake_tables= set(old_datalake_tables)

    diff = src_tables - old_datalake_tables

    lostbench_tables = os.listdir(LOSTBENCH_PATH)
    lostbench_tables = [table for table in lostbench_tables if table != '.DS_Store']
    lostbench_tables = set(lostbench_tables)

    diff2 = lostbench_tables - diff

    print(diff2)
    print(len(diff2))



    
