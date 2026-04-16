'''
For the Generation of Lookup dictionaries for
1. ANES VAR CODE -> ANES LABELS
2. ANES VAR CODE -> ANES QUESTIONS
3. clean then by normalizing whitespaces- \t, \n, \s+ 
'''
import json
import pandas as pd
import re

def main():
    with open(CUMULATIVE_PATH, "r", encoding = "utf-8") as f:
        cumulative = json.load(f)

    cumulative_dict = {}

    for vcf in cumulative:
        for k,v in vcf.items():
            cumulative_dict[k] = v

    # output = {}
    labels = {}
    questions = {}

    for year in years:
        source_path = f"../../output/{year}/codebook_final.json"
        
        try:
            with open(source_path, "r", encoding = "utf-8") as f:
                source = json.load(f)
            source_dict = {}
            
            for item in source:
                for key, value in item.items():
                    if year in ['1996', '1998', '2000']: 
                        source_dict[f"{key[4:]}"] = value
                    elif year in ['2002', '2004', '2008', '2012', '2016', '2020']:
                        source_dict[f"{key}"] = value
                    else:
                        source_dict[f"V{key[4:]}"] = value

            for key, value in source_dict.items():
                if key in labels:
                    '''
                    Use the information from this print to decide how to manage repeats like these in dataset V2.0
                    '''
                    print(f"{year}-->{key}-->{value.get('LABEL', value)}")
                if year == '2012':
                    labels[key] = value["Label:"]
                    questions[key] = value.get("Question:", value["Label:"])
                elif year == '2016':
                    labels[key] = value["Description"]
                    if "Not applicable" not in value.get("Question:", ""):
                        questions[key] = value.get("Question:", value["Description"])
                    else:
                        questions[key] = value["Description"]
                elif year == '2020':
                    labels[key] = value["Description"]
                    questions[key] = value.get("Question", value["Description"])
                else:
                    labels[key] = value["LABEL"]
                    questions[key] = value.get("QUESTION", value["LABEL"])
        
        except Exception as e:
            print("Error: ", e, year, key)
            print("Error type:", type(e).__name__)
        
    return labels, questions

if __name__ == "__main__":

    years = ['1948', '1952', '1956', '1958', '1960', '1962', '1964', '1966',
        '1968', '1970', '1972', '1974', '1976', '1978', '1980', '1982', 
        '1984', '1986', '1988', '1990', '1994', '1996','1998', '2000', 
        '2002', '2004','2008', '2012', '2016', '2020'] 
        #1996 does not have categories

    '''
    Change these paths based on your codebase directory
    '''
    LABELS_LOOKUP_OUTPUT_PATH = "../groundtruth/labels_lookup.json"
    QUESTIONS_LOOKUP_OUTPUT_PATH = "../groundtruth/questions_lookup.json"

    '''
    Change it to the path where cumulative codebook's codebook_final.json is present
    '''
    CUMULATIVE_PATH = f"../../output/cumulative/codebook_final.json"

    labels, questions = main()
    
    cleaned_labels = {
        k: re.sub(r'\s+', ' ', v).strip()
        for k, v in labels.items()
    }

    cleaned_questions = {
        k: re.sub(r'\s+', ' ', v).strip()
        for k, v in questions.items()
    }

    with open(LABELS_LOOKUP_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(cleaned_labels, f_out, indent=2, ensure_ascii=False)
    print(f"Saved labels lookup to {LABELS_LOOKUP_OUTPUT_PATH} successfully")

    with open(QUESTIONS_LOOKUP_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(cleaned_questions, f_out, indent=2, ensure_ascii=False)
    print(f"Saved questions lookup to {QUESTIONS_LOOKUP_OUTPUT_PATH} successfully")
