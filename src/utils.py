from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor


def get_timestamp() -> str:
    """Returns current timestamp to append to files

    Returns: the timestamp string
    """
    current_timestamp = datetime.now()
    processed_timestamp = (
        str(current_timestamp)[:-7].replace(" ", "_").replace(":", "").replace("-", "") + "_"
    )

    return processed_timestamp


def find_hyper_glycemia_hours(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find the hours where blood glucose goes above 125 mg/dl

    Args:
        df : the dataframe holding the time series observations for a patient

    Returns: list of (start, end) tuples
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
        df : the dataframe holding the time series observations for a patient

    Returns : list of (start, end) tuples
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


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float("-inf"), diagonal=1)


def get_indices_for_sequence(
    time_series_len: int, window_size: int, step_size: int
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Produce all the start and end index positions that is needed to produce
    the pateint-specific sub-sequences.
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences.

    Args:
        data : The dataframe we want to slice
        window_size : The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50
                           time steps, window_size = 100+50 = 150
        step_size : Size of each step as the data sequence is traversed
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size],
                         and the next will be [1:window_size].
    Return:
        indices: a list of tuples
        num_samples: number of input output samples
    """

    stop_position = time_series_len - 1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []
    num_samples = 0

    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size
        num_samples += 1

    return indices, num_samples


def get_patient_indices(
    data: List[torch.tensor],
    input_seq_len: int,
    forecast_len: int,
    step_size: int,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """
    Produce all the start and end index positions that is needed to produce

    Args:
        data : The entire data patient we want to slice
        input_seq_len : the size of the sequence that will be input to the model
        forecast_len : the size of the sequence that the model forecasts
        step_size : Size of each step as the data sequence is traversed
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size],
                         and the next will be [1:window_size].
    Return:
        indices: a list of tuples
        num_samples: total number of input, output samples
    """

    window_size = input_seq_len + forecast_len

    indices = []
    total_samples = 0

    for patient in data:
        time_series_len = len(patient)
        patient_indices, patient_num_samples = get_indices_for_sequence(
            time_series_len, window_size, step_size
        )
        indices.append(patient_indices)
        total_samples += patient_num_samples

    return indices, total_samples


def read_data(fpath_data: Path, timestamp_col_name: str = "timestamp") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object
    Args:
        data_dir: str or Path object specifying the path to the csv file
        timestamp_col_name: str, the name of the column or named index
                            containing the timestamps
    """

    print("Reading file in {}".format(fpath_data))

    data = pd.read_csv(
        fpath_data,
        parse_dates=[timestamp_col_name],
        index_col=[timestamp_col_name],
        infer_datetime_format=True,
        low_memory=False,
    )

    # Make sure all "n/e" values have been removed from df.
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data


def is_ne_in_df(df: pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """

    for col in df.columns:

        true_bool = df[col] == "n/e"

        if any(true_bool):
            return True

    return False


def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes("float").columns

    icols = df.select_dtypes("integer").columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")

    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

    return df
