import pandas as pd

METRICS_FILE = "DDM_LLM_sm_top20.csv"
RETRIES_FILE = "DDM_LLM_retries_top20.csv"


metrics = pd.read_csv(METRICS_FILE)
retries = pd.read_csv(RETRIES_FILE)

print(metrics['One2One_RecallAtSizeofGroundTruth'].mean())
print((retries['num_retries'] != 0).sum())