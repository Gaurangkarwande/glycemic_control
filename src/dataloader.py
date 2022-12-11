from logging import Logger
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from src.dataset import (
    GraphDataset,
    df_to_patient_tensors,
    get_adjacency_coo,
    get_normalizing_scaler,
    stack_dataset_featues_target,
    get_adjacency_matrix,
    SCALER_TYPE,
)
from src.utils import get_patient_indices
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


def get_dataloaders(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    df_dag: Optional[pd.DataFrame],
    input_variables: List[str],
    target_variable: str,
    enc_seq_len: int,
    output_sequence_length: int,
    step_size: int,
    batch_size: int,
    logger: Optional[Logger] = None,
    verbose: bool = True,
    normalize_target: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[SCALER_TYPE]]:
    """Convert given dataframes into ML dataset

    Args:
        df_train: training data
        df_test: training data
        df_test: training data
        df_dag: the dag
        input_variable: the names of all covariates
        target_variable: the name of the target col in dataframe
        enc_seq_len: the number of time steps given as input
        output_sequence_length: the number of time steps predicted
        step_size: the window time step
        batch_size: the batch size for batching
        logger: the logger object
        verbose: whether to print details. Default is True.
        normalize_target: whether to normalize the target variable. Default is False.

    Returns:
        the train, valid, and test dataloader
        the scaler for target variable
    """

    # df to patient tensor
    scaler_x = get_normalizing_scaler(df_train[input_variables].values)
    scaler_y = None
    if normalize_target:
        scaler_y = get_normalizing_scaler(df_train[target_variable].values)
    else:
        logger.info("Not Normalizing glucose values")
        if verbose:
            print("Not Normalizing glucose values")

    X_train, y_train = df_to_patient_tensors(
        df_train,
        feature_cols=input_variables,
        target_col=target_variable,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )
    X_valid, y_valid = df_to_patient_tensors(
        df_valid,
        feature_cols=input_variables,
        target_col=target_variable,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )
    X_test, y_test = df_to_patient_tensors(
        df_test,
        feature_cols=input_variables,
        target_col=target_variable,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
    )

    # get subsequence indices
    indices_train, num_samples_train = get_patient_indices(
        y_train, input_seq_len=enc_seq_len, forecast_len=output_sequence_length, step_size=step_size
    )
    indices_valid, num_samples_valid = get_patient_indices(
        y_valid, input_seq_len=enc_seq_len, forecast_len=output_sequence_length, step_size=step_size
    )
    indices_test, num_samples_test = get_patient_indices(
        y_test, input_seq_len=enc_seq_len, forecast_len=output_sequence_length, step_size=step_size
    )

    logger.info(
        f"Number of training samples: {num_samples_train}"
        f" \nNumber of valid samples: {num_samples_valid}"
        f" \nNumber of test samples: {num_samples_test}"
    )

    if verbose:
        print(
            f"Number of training samples: {num_samples_train}"
            f" \nNumber of valid samples: {num_samples_valid}"
            f" \nNumber of test samples: {num_samples_test}"
        )

    # create datasets

    dataset_train = GraphDataset(
        data=X_train,
        labels=y_train,
        indices=indices_train,
        num_samples=num_samples_train,
        enc_seq_len=enc_seq_len,
        target_seq_len=output_sequence_length,
    )
    dataset_valid = GraphDataset(
        data=X_valid,
        labels=y_valid,
        indices=indices_valid,
        num_samples=num_samples_valid,
        enc_seq_len=enc_seq_len,
        target_seq_len=output_sequence_length,
    )
    dataset_test = GraphDataset(
        data=X_test,
        labels=y_test,
        indices=indices_test,
        num_samples=num_samples_test,
        enc_seq_len=enc_seq_len,
        target_seq_len=output_sequence_length,
    )

    # create initial dataloaders

    X_train, y_train = stack_dataset_featues_target(dataset_train)
    X_valid, y_valid = stack_dataset_featues_target(dataset_valid)
    X_test, y_test = stack_dataset_featues_target(dataset_test)

    # create adjacency matrix

    adj_matrix = get_adjacency_matrix(df_dag)
    if adj_matrix is None:
        adj_matrix = np.ones(shape=(len(input_variables), len(input_variables)))
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 0
        logger.info("Running with fully connected DAG")

    edge_index, edge_weights = get_adjacency_coo(adj_matrix)
    # adj_matrix = torch.tensor(adj_matrix)
    # edge_index = adj_matrix.nonzero().t().contiguous()
    # edge_weights = torch.ones(edge_index.shape[1])

    logger.info(
        f"ADJACENCY MATRIX - Number of edges is {len(edge_weights)}"
    )  # bidirectional edge counted as two distinct edge

    dataloader_train_temp = StaticGraphTemporalSignal(
        edge_index=edge_index, edge_weight=edge_weights, features=X_train, targets=y_train
    )
    dataloader_valid_temp = StaticGraphTemporalSignal(
        edge_index=edge_index, edge_weight=edge_weights, features=X_valid, targets=y_valid
    )
    dataloader_test_temp = StaticGraphTemporalSignal(
        edge_index=edge_index, edge_weight=edge_weights, features=X_test, targets=y_test
    )

    # create torch_geometic dataloader for batched operation

    dataloader_train = DataLoader(dataset=list(dataloader_train_temp), batch_size=batch_size)
    dataloader_valid = DataLoader(dataset=list(dataloader_valid_temp), batch_size=batch_size)
    dataloader_test = DataLoader(dataset=list(dataloader_test_temp), batch_size=batch_size)

    return dataloader_train, dataloader_valid, dataloader_test, scaler_y
