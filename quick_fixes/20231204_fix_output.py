from lib.io import read_data
from lib.utils import setup_log
from lib.config import MyConfig
from configlist import config_list

skip_if_exist = True
skip_if_exist = False

def main():
    for config_files in config_list:
        config = MyConfig(file_paths=config_files)
        
        for dp in config.data_paths:
            df = read_data(dp.result_file)
            for i, row in df.iterrows():
                agreement = calc_agreement(
                    ground_truth_score=row["ETS Score"],
                    llm_score=row["LLM Score"],
                    integer_score_only=config.integer_score_only,
                )
                for k, v in agreement.items():
                    df.loc[i, k] = v
            df.to_excel(dp.result_file, index=False)
            # break
        # break

def calc_agreement(ground_truth_score: float | int, llm_score: float | int, integer_score_only: bool) -> dict:
    diff = llm_score - ground_truth_score
    is_agree = abs(diff) < 0.51

    if integer_score_only:
        agreement_type = "0"
        if abs(diff) < 0.01:
            agreement_type = "2"
        elif abs(diff) < 0.51:
            agreement_type = "1-high" if diff > 0 else "1-low"
    else:
        agreement_type = "0"
        if abs(diff) < 0.01:
            agreement_type = "abs"
        elif abs(diff) < 0.51:
            agreement_type = "adj"

    return {
        "Agreement or not": is_agree,
        "Agreement type": agreement_type,
    }

if __name__ == "__main__":
    setup_log()
    main()
