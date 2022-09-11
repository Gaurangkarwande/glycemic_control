from datetime import datetime
from typing import List, Tuple

import pandas as pd


def get_timestamp():
    """Returns current timestamp to append to files"""
    current_timestamp = datetime.now()
    processed_timestamp = (
        str(current_timestamp)[:-7].replace(" ", "_").replace(":", "").replace("-", "") + "_"
    )

    return processed_timestamp


def find_hyper_glycemia_hours(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find the hours where blood glucose goes above 125 mg/dl

    Args:
        df (pd.DataFrame) : the dataframe holding the time series observations for a patient

    Returns:
        List[Tuple[int, int]] : list of (start, end) tuples
    """

    prev_hyper_timestamp = None
    hours_list = []
    for row in df.itertuples():
        if row.glucose < 125:
            if prev_hyper_timestamp is not None:
                hours_list.append((prev_hyper_timestamp, row.hr))
                prev_hyper_timestamp = None
            else:
                continue
        elif row.glucose > 125:
            if prev_hyper_timestamp is not None:
                continue
            else:
                prev_hyper_timestamp = row.hr
    return hours_list


def find_hypo_glycemia_hours(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find the hours where blood glucose goes below 70 mg/dl

    Args:
        df (pd.DataFrame) : the dataframe holding the time series observations for a patient

    Returns:
        List[Tuple[int, int]] : list of (start, end) tuples
    """

    prev_hypo_timestamp = None
    hours_list = []
    for row in df.itertuples():
        if row.glucose > 70:
            if prev_hypo_timestamp is not None:
                hours_list.append((prev_hypo_timestamp, row.hr))
                prev_hypo_timestamp = None
            else:
                continue
        elif row.glucose < 70:
            if prev_hypo_timestamp is not None:
                continue
            else:
                prev_hypo_timestamp = row.hr
    return hours_list
