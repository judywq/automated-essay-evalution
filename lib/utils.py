import os
import pandas as pd
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from setting import DEFAULT_LOG_LEVEL

import logging
logger = logging.getLogger(__name__)


def get_date_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def calc_agreement(ground_truth_score: float | int, llm_score: float | int, integer_score_only: bool) -> dict:
    """Calculate the agreement between ground truth score and LLM Score

    Args:
        ground_truth_score (float|int): ground truth score
        llm_score (float|int): score from LLM model

    Returns:
        dict: a dict of agreement type and whether the two scores agree
        {
            "Agreement or not": bool,
            "Agreement type": "0" | "1-high" | "1-low" | "2"
        }
    """
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

def calc_metrics_dict(result_fn, integers_only) -> dict:
    if not os.path.exists(result_fn):
        return {}
    df_data = pd.read_excel(result_fn)
    res = {}

    # All kinds of metrics
    def rmse_metric(y_true, y_pred):
        # Calculate Root Mean Squared Error (RMSE)
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def kappa_metric(y_true, y_pred):
        y_true_labels = (y_true * 2).astype(int)
        y_pred_labels = (y_pred * 2).astype(int)
        # Calculate Cohen's kappa
        return cohen_kappa_score(y_true_labels, y_pred_labels, weights='quadratic')
    
    def get_metrics(metric_func, df, filter_form=None):
        if filter_form:
            df = df[df['essay_form_id'] == filter_form]
        
        if df.empty:
            return "-"
        actual_values = df['ETS Score']
        if 'LLM Score' in df.columns:
            predicted_values = df['LLM Score']
        else:
            predicted_values = df['GPT Score']
        return metric_func(actual_values, predicted_values)

    for metric_name, metric_func in [
        ('RMSE', rmse_metric),
        ('Kappa', kappa_metric),
    ]:
        res[metric_name] = get_metrics(metric_func, df_data)
        res[f'{metric_name}-P1'] = get_metrics(metric_func, df_data, filter_form=1)
        res[f'{metric_name}-P2'] = get_metrics(metric_func, df_data, filter_form=2)

    # All kinds of agreement
    def get_agreement_int(df, res, filter_form=None):
        suffix = ''
        if filter_form:
            df = df[df['essay_form_id'] == filter_form]
            suffix = f'-P{filter_form}'
        res['i-total' + suffix] = df['Agreement type'].isin(['2', '1-high', '1-low']).mean()
        res['2' + suffix] = df['Agreement type'].isin(['2']).mean()
        res['1T' + suffix] = df['Agreement type'].isin(['1-high', '1-low']).mean()
        res['1H' + suffix] = df['Agreement type'].isin(['1-high']).mean()
        res['1L' + suffix] = df['Agreement type'].isin(['1-low']).mean()

    def get_agreement_float(df, res, filter_form=None):
        suffix = ''
        if filter_form:
            df = df[df['essay_form_id'] == filter_form]
            suffix = f'-P{filter_form}'
        res['f-total' + suffix] = df['Agreement type'].isin(['abs', 'adj']).mean()
        res['abs' + suffix] = df['Agreement type'].isin(['abs']).mean()
        res['adj' + suffix] = df['Agreement type'].isin(['adj']).mean()
    
    if integers_only:
        for form_id in [None, 1, 2]:
            get_agreement_int(df_data, res, filter_form=form_id)
        # res = {
        #     **res,
        #     'f-total': '',
        #     'abs': '',
        #     'adj': '',
        # }
    else:
        for form_id in [None, 1, 2]:
            get_agreement_float(df_data, res, filter_form=form_id)
        # res = {
        #     'i-total': '',
        #     '2': '',
        #     '1T': '',
        #     '1H': '',
        #     '1L': '',
        #     **res,
        # }
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
