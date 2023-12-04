import os
import pandas as pd
import datetime
from setting import DEFAULT_LOG_LEVEL

import logging
logger = logging.getLogger(__name__)


def get_date_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def calc_agreement(ground_truth_score: float | int, gpt_score: float | int) -> dict:
    """Calculate the agreement between ground truth score and GPT score

    Args:
        ground_truth_score (float|int): ground truth score
        gpt_score (float|int): score from GPT

    Returns:
        dict: a dict of agreement type and whether the two scores agree
        {
            "Agreement or not": bool,
            "Agreement type": "0" | "1-high" | "1-low" | "2"
        }
    """
    diff = gpt_score - ground_truth_score
    is_agree = abs(diff) < 0.51

    agreement_type = "0"
    if abs(diff) < 0.01:
        agreement_type = "2"
    elif abs(diff) < 0.51:
        agreement_type = "1-high" if diff > 0 else "1-low"

    return {
        "Agreement or not": is_agree,
        "Agreement type": agreement_type,
    }

def calc_and_write_success_rate(result_fn):
    df_data = pd.read_excel(result_fn)
    content = ""
    for col_name in ["Agreement or not"]:
        if col_name in df_data.columns:
            rate = df_data[col_name].sum() / len(df_data)
            content += f"{col_name}: {rate}\n"
    
    with open(result_fn + ".txt", "a") as f:
        f.write(content)
    print(content)

def calc_success_rate(result_fn):
    df_data = pd.read_excel(result_fn)
    col_name = ["Agreement or not"]
    rate = df_data[col_name].sum() / len(df_data)
    return rate.iloc[0]

def calc_success_rate_dict(result_fn, integers_only) -> dict:
    df_data = pd.read_excel(result_fn)
    res = {}
    if integers_only:
        res['int-total'] = df_data['Agreement type'].isin(['2', '1-high', '1-low']).mean()
        res['int-2'] = df_data['Agreement type'].isin(['2']).mean()
        res['int-1-total'] = df_data['Agreement type'].isin(['1-high', '1-low']).mean()
        res['int-1-high'] = df_data['Agreement type'].isin(['1-high']).mean()
        res['int-1-low'] = df_data['Agreement type'].isin(['1-low']).mean()
        res = {
            **res,
            'float-total': '',
            'float-absolute': '',
            'float-adjacent': '',
        }
    else:
        res['float-total'] = df_data['Agreement type'].isin(['2', '1-high', '1-low']).mean()
        res['float-absolute'] = df_data['Agreement type'].isin(['2']).mean()
        res['float-adjacent'] = df_data['Agreement type'].isin(['1-high', '1-low']).mean()
        res = {
            'int-total': '',
            'int-2': '',
            'int-1-total': '',
            'int-1-high': '',
            'int-1-low': '',
            **res,
        }
    return res


def setup_log(level=None, log_path='./log/txt', need_file=True):
    if not level:
        level = logging.getLevelName(DEFAULT_LOG_LEVEL)
    if not os.path.exists(log_path):
        os.makedirs(log_path)    
        
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s: %(message)s")
    
    handlers = []
    if need_file:
        filename = get_date_str()
        file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, filename))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level=level)
    handlers.append(console_handler)

    # https://stackoverflow.com/a/11111212
    logging.basicConfig(level=logging.DEBUG,
                        handlers=handlers)
