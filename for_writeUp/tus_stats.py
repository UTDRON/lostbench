import pandas as pd
import os
import json

def calc_table_stats(benchmark, separator):

    for type in [DATALAKE_PATH, QUERY_PATH]:
        total_rows = 0
        total_cols = 0
        file_count = 0
        
        for file in os.listdir(type):
            if file.endswith(".csv"):
                path = os.path.join(type, file)

                try:

                    df = pd.read_csv(path, sep=separator)
                    rows, cols = df.shape

                    total_rows += rows
                    total_cols += cols
                    file_count += 1
                
                except Exception as e:
                    print(e, file)
                    continue
        
        avg_rows = total_rows / file_count
        avg_cols = total_cols / file_count

        print(f"For {benchmark}:{type}")
        print("Number of files:", file_count)
        print("Average rows:", avg_rows)
        print("Average columns:", avg_cols)
        print("-------------------")


def calc_null_percentage(benchmark, separator):

    for type in [DATALAKE_PATH, QUERY_PATH]:
        missing_ratios = 0
        file_count = 0

        for file in os.listdir(type):
            if file.endswith(".csv"):
                path = os.path.join(type, file)

                try:
                    df = pd.read_csv(path,sep=separator)
                    if df.size > 0:
                        ratio = df.isna().sum().sum() / df.size
                    else:
                        ratio = 1
                    missing_ratios += ratio
                    # print(missing_ratios)

                    file_count += 1


                except Exception as e:
                    print(e, file)
                    continue
        
        print(f"For {benchmark}:{type}")
        print(f"Percentage Null: {missing_ratios/file_count}")


def calc_numeric_percentage(benchmark, separator):

    for type in [DATALAKE_PATH, QUERY_PATH]:
        num_numeric_cols = 0
        total_cols = 0

        for file in os.listdir(type):
            if file.endswith(".csv"):
                path = os.path.join(type, file)

                try:
                    df = pd.read_csv(path,sep=separator)
                    num_numeric_cols += df.select_dtypes(include='number').shape[1]
                    total_cols += df.shape[1]

                except Exception as e:
                    print(e, file)
                    continue
        
        print(f"For {benchmark}:{type}")
        print(f"Percentage Numeric: {num_numeric_cols/total_cols}")


        
if __name__ == "__main__":
    
    '''s
    For LoStBench
    '''
    DATALAKE_PATH = "../LoStBench/datalake/"
    QUERY_PATH = "../LoStBench/query/"

    # calc_table_stats("LoStBench",",")
    # calc_null_percentage("LoStBench",",")
    # calc_numeric_percentage("LoStBench",",")

    '''
    For SANTOS SMALL
    '''
    DATALAKE_PATH = "../../santos/benchmark/santos_benchmark/datalake/"
    QUERY_PATH = "../../santos/benchmark/santos_benchmark/query/"

    # calc_table_stats("SANTOS SMALL",",")
    # calc_null_percentage("SANTOS SMALL",",")
    # calc_numeric_percentage("SANTOS SMALL",",")

    '''
    For SANTOS LARGE
    '''
    DATALAKE_PATH = "../../santos/benchmark/real_tables_benchmark/datalake/"
    QUERY_PATH = "../../santos/benchmark/real_tables_benchmark/query/"

    # calc_table_stats("SANTOS LARGE",",")
    # calc_null_percentage("SANTOS LARGE",",")
    # calc_numeric_percentage("SANTOS LARGE",",")

    '''
    For UGEN V2
    '''
    DATALAKE_PATH = "../../gen/data/ugen_v2/datalake/"
    QUERY_PATH = "../../gen/data/ugen_v2/query/"

    # calc_table_stats("UGEN V2",";")
    # calc_null_percentage("UGEN V2",";")
    calc_numeric_percentage("UGEN V2",";")




    
