import pandas as pd

if __name__ == "__main__":

    # ====== For Match Pairs =======

    # INPUT_PATH = "sm_lostbench_qn_match_output.csv"
    # OUTPUT_MAGNETO_PATH = "magneto_sm_lostbench_qn_match_output.csv"
    # OUTPUT_MAGNETOGPT_PATH = "magnetogpt_sm_lostbench_qn_match_output.csv"

    # input_df = pd.read_csv(INPUT_PATH)

    # magneto_df = input_df[input_df["method"] == "Magneto"]
    # magnetogpt_df = input_df[input_df["method"] == "MagnetoGPT"]

    # magneto_df.to_csv(OUTPUT_MAGNETO_PATH, index=False)
    # print(f"saved filtered magneto to {OUTPUT_MAGNETO_PATH} sucecessfully!!")

    # magnetogpt_df.to_csv(OUTPUT_MAGNETOGPT_PATH, index=False)
    # print(f"saved filtered magnetogpt to {OUTPUT_MAGNETOGPT_PATH} sucecessfully!!")


    # ====== For Metrics =======

    INPUT_METRICS_FILE = "../ANES_450_default_e-s_qns_lbl_new_prompt.csv"
    MAGNETO_OUTPUT_METRICS_PATH = "../output_metrics/magneto_sm_lostbench_qn_metrics.csv"
    MAGNETOGPT_OUTPUT_METRICS_PATH = "../output_metrics/magnetogpt_sm_lostbench_qn_metrics.csv"

    input_df = pd.read_csv(INPUT_METRICS_FILE)

    magneto_df = input_df[input_df["method"] == "Magneto"]
    magnetogpt_df = input_df[input_df["method"] == "MagnetoGPT"]

    magneto_df.to_csv(MAGNETO_OUTPUT_METRICS_PATH, index=False)
    print(f"saved filtered magneto to {MAGNETO_OUTPUT_METRICS_PATH} sucecessfully!!")

    magnetogpt_df.to_csv(MAGNETOGPT_OUTPUT_METRICS_PATH, index=False)
    print(f"saved filtered magnetogpt to {MAGNETOGPT_OUTPUT_METRICS_PATH} sucecessfully!!")
