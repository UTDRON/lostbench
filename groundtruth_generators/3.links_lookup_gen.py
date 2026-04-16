'''
For the Generation of Lookup dictionary for
ANES Source Var(V......) -> List of ANES Cumulative Vars(VCF......)
'''
import json
import pandas as pd

def main():
    with open(CUMULATIVE_PATH, "r", encoding = "utf-8") as f:
        cumulative = json.load(f)

    cumulative_dict = {}

    for vcf in cumulative:
        for k,v in vcf.items():
            cumulative_dict[k] = v

    with open(LABELS_LOOKUP_PATH, "r", encoding = "utf-8") as f:
        source = json.load(f)

    output = {}

    for year in years:
        try:
            for key, value in cumulative_dict.items():
                if "Sources" in value:
                    if year in value["Sources"]:
                        for var in value["Sources"][year]:
                            if var in source:
                                if var not in output:
                                    output[var] = [key]
                                else:
                                    output[var].append(key)
        
        except Exception as e:
            print("Error: ", e)
            print("Error type:", type(e).__name__)
    
    return output

if __name__ == "__main__":
    years = ['1948', '1952', '1956', '1958', '1960', '1962', '1964', '1966',
        '1968', '1970', '1972', '1974', '1976', '1978', '1980', '1982', 
        '1984', '1986', '1988', '1990', '1994', '1996','1998', '2000', 
        '2002', '2004','2008', '2012', '2016', '2020'] 
        #1996 does not have categories

    '''
    Change these paths based on your codebase directory
    '''
    LABELS_LOOKUP_PATH = "../groundtruth/labels_lookup.json" #labels lookup has what we need now as reqd information
    LINKS_LOOKUP_OUTPUT_PATH = "../groundtruth/links_lookup.json"

    '''
    Change it to the path where cumulative codebook's codebook_final.json is present
    '''
    CUMULATIVE_PATH = f"../../output/cumulative/codebook_final.json"

    output = main()

    with open(LINKS_LOOKUP_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=2, ensure_ascii=False)
    print(f"Saved V... -> [VCFs] lookup to {LINKS_LOOKUP_OUTPUT_PATH} successfully")
    


