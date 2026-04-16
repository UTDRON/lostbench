import json
import pandas as pd
import random

if __name__ == "__main__":

    random.seed(42)

    TEST_FILE = "test_case.json"
    TEST_FILE_RANDOM = "test_case_random.json"

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_file = json.load(f)

    for key in test_file:
        random.shuffle(test_file[key])

    with open(TEST_FILE_RANDOM, 'w', encoding='utf-8') as f_out:
        json.dump(test_file, f_out, indent=2, ensure_ascii=False)