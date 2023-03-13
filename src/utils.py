from datetime import datetime
from typing import List, Tuple

import numpy as np
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


def find_num_edges(adj_matrix: np.ndarray) -> int:
    """Find the number of edges from adjacency matrix

    Args:
        adj_matrix: the adjacency matrix as a numpy array

    Returns: the number of edges
    """

    num_edges = np.where(adj_matrix > 0, 1, 0).sum()
    return num_edges


def aggregate_partial_dags(partial_dags: List[np.ndarray], majority_threshold: int) -> np.ndarray:
    """Aggregate the partial dags by majority voting.
    Algorithm:
        1. Select an edge, say edge from node i to j. Direction is important.
        2. Find number of partial dags having this particular edge
        3. If the number is greater than the threshold then the aggregated dag has this edge.
        Assign the edge weight in aggregated dag to the maximum edge weight for this edge in all
        partial dags.

    Args:
        partial_dags: the list of 2d partial dags to aggregate
        majority_threshold: the threshold value used in step 3

    Returns: the aggregated dag
    """

    assert len(partial_dags) > 0, "The list of partial dags is empty"
    partial_dag_set = np.stack(partial_dags, axis=2)
    dag_shape = partial_dags[0].shape

    aggregated_dag = np.zeros(shape=dag_shape)

    for row in range(dag_shape[0]):
        for col in range(dag_shape[1]):
            sequence = partial_dag_set[row, col]
            num_votes = np.where(sequence > 0, 1, 0).sum()
            aggregated_dag[row, col] = sequence.max() if num_votes > majority_threshold else 0

    return aggregated_dag


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


def dag_fully_connected(num_nodes: int, add_self_loops: int) -> np.ndarray:
    """Create a fully connected dag

    Args:
        num_nodes: number of nodes in the graph
        add_self_loops: whether to add self loops

    Returns: the adjacency matrix
    """
    adj_matrix = np.ones((num_nodes, num_nodes))
    if not add_self_loops:
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 0

    return adj_matrix


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=4, min_lr=1e-6, factor=0.3, verbose=True):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        :verbose: whether to print details
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=verbose,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

    def get_final_lr(self):
        return self.lr_scheduler.state_dict()


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=11, min_delta=1e-5, verbose=True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :verbose: whether to print details
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("INFO: Early stopping")
                self.early_stop = True
