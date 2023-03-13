from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from random import sample, shuffle
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.nn.utils.rnn import pad_sequence
from src.constants import INPUT_COVARIATES, TARGET_COL
from src.utils import dag_fully_connected


SCALER_TYPE = RobustScaler


class GraphClassificationDataset(Dataset):
    """The Dataset class for graph classification"""

    def __init__(
        self,
        data: torch.Tensor,
        seq_lens: List[int],
        target_labels: List[int],
        init_adj_mat: np.ndarray,
    ) -> None:
        """
        Args:
            data: the padded time series data
            seq_lens: the original unpadded sequence lengths
            target: the class labels
        """
        super().__init__()
        selected_sample_ids = balance_dataset(target_labels)
        self.data = data[selected_sample_ids]
        self.seq_lens = [seq_lens[i] for i in selected_sample_ids]
        self.target_labels = [target_labels[i] for i in selected_sample_ids]
        self.init_adj_mat = init_adj_mat
        self.init_edge_index, self.init_edge_weight = dense_to_sparse(torch.as_tensor(init_adj_mat))

    def __len__(self) -> int:
        return len(self.target_labels)

    def __getitem__(self, index):
        return Data(
            x=self.data[index].T,
            y=torch.tensor(self.target_labels[index]),
            seq_lens=self.seq_lens[index],
            edge_index=self.init_edge_index,
            edge_attr=self.init_edge_weight.unsqueeze(dim=1),
        )


class LSTMClassificationDataset(Dataset):
    """The Dataset class for graph classification"""

    def __init__(
        self,
        data: torch.Tensor,
        seq_lens: List[int],
        target_labels: List[int],
        init_adj_mat: np.ndarray,
    ) -> None:
        """
        Args:
            data: the padded time series data
            seq_lens: the original unpadded sequence lengths
            target: the class labels
        """
        super().__init__()
        selected_sample_ids = balance_dataset(target_labels)
        self.data = data[selected_sample_ids]
        self.seq_lens = [seq_lens[i] for i in selected_sample_ids]
        self.target_labels = [target_labels[i] for i in selected_sample_ids]
        self.init_adj_mat = init_adj_mat
        self.init_edge_index, self.init_edge_weight = dense_to_sparse(torch.as_tensor(init_adj_mat))

    def __len__(self) -> int:
        return len(self.target_labels)

    def __getitem__(self, index):
        return self.data[index].T, torch.tensor(self.target_labels[index]), self.seq_lens[index]


def balance_dataset(target_labels: List[int]) -> List[int]:
    """Balance the classes

    Args:
        target_labels: list of target labels

    Returns: indices of balanced dataset
    """
    negative_ids = []
    positive_ids = []

    for id, label in enumerate(target_labels):
        if label == 0:
            negative_ids.append(id)
        else:
            positive_ids.append(id)
    num_samples = min(len(positive_ids), len(negative_ids))
    balanced_ids = sample(positive_ids, num_samples) + sample(negative_ids, num_samples)
    shuffle(balanced_ids)
    return balanced_ids


def get_adjacency_matrix(df_dag: Optional[pd.DataFrame]) -> np.ndarray:
    """Get adjacency matrix from dataframe

    Args:
        df_dag: the dag dataframe

    Returns: the adjacency matrix
    """

    if df_dag is None:
        return None
    adj_matrix = df_dag[INPUT_COVARIATES + TARGET_COL].to_numpy()
    # this is an upper triangular matrix
    for row in range(adj_matrix.shape[0]):
        for col in range(row):
            adj_matrix[row, col] = adj_matrix[col, row]
    assert (adj_matrix == adj_matrix.T).all(), "Adjacency matrix not symmetric"
    return adj_matrix


def get_adjacency_coo(adj_matrix: np.ndarray) -> np.ndarray:
    """Convert adjacency matrix to COO format for pytorch_geometric

    Args:
        adj_matrix: the adjacency matrix to convert

    Returns: adjacency matrix in COO format
    """

    source = []
    destination = []
    edge_weights = []

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                source.append(i)
                destination.append(j)
                edge_weights.append(adj_matrix[i, j])

    coo_adjacency = np.stack([source, destination])
    edge_weights = np.array(edge_weights)
    return coo_adjacency, edge_weights


def get_normalizing_scaler(X_train: np.array) -> SCALER_TYPE:
    """Get the preprocessing scaler that normalizes the feature values

    Args:
        X_train : the training dataset

    Returns: the fitted scaler. Could be one of MinMaxScaler or StandardScaler
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def df_to_patient_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    scaler: SCALER_TYPE = None,
) -> Tuple[List[torch.tensor], List[int], List[int]]:
    """Convert the dataframe to tensor data

    Args:
        df : the data in dataframe form
        feature_cols : the list of covariates we want to use as features in the multivariate
            timeseries
        target_col : the name of the column having glucose values
        scaler : the preprocessing scaler used to normalize the data

    Returns:
        patient_tensor: the patient time series as tensor
        patient_seq_lens: the lenght of each patient time series
        target_labels: the patient class
    """

    patient_features = []
    target_labels = []
    patient_seq_lens = []
    df_patient_group = df.groupby(by="subject_id")
    for _, df_patient in df_patient_group:
        features = df_patient[feature_cols].to_numpy()
        if scaler is not None:
            features = scaler.transform(features)

        patient_features.append(torch.as_tensor(features, dtype=torch.float32))
        patient_seq_lens.append(features.shape[0])
        target_labels.append(df_patient[target_col].unique()[0])

    return patient_features, patient_seq_lens, target_labels


def build_classification_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: List[str],
) -> Tuple[GraphClassificationDataset, GraphClassificationDataset, GraphClassificationDataset]:
    """Build the graph classification dataste from given dataframe

    Args:
        df_train: the training dataframe
        df_test: the validation dataframe
        df_val: the testing dataframe
        feature_cols: list of input variable names
        target_col: the target variable

    Returns: the build dataset
    """

    scaler = get_normalizing_scaler(df_train[feature_cols].values)

    patient_tensors_train, patient_seq_lens_train, target_labels_train = df_to_patient_tensors(
        df_train, feature_cols=feature_cols, target_col=target_col, scaler=scaler
    )
    patient_tensors_train = pad_sequence(patient_tensors_train, batch_first=True)
    init_adj_mat = dag_fully_connected(num_nodes=len(feature_cols), add_self_loops=False)

    dataset_train = GraphClassificationDataset(
        data=patient_tensors_train,
        seq_lens=patient_seq_lens_train,
        target_labels=target_labels_train,
        init_adj_mat=init_adj_mat,
    )

    patient_tensors_val, patient_seq_lens_val, target_labels_val = df_to_patient_tensors(
        df_val, feature_cols=feature_cols, target_col=target_col, scaler=scaler
    )
    patient_tensors_val = pad_sequence(patient_tensors_val, batch_first=True)
    init_adj_mat = dag_fully_connected(num_nodes=len(feature_cols), add_self_loops=False)

    dataset_val = GraphClassificationDataset(
        data=patient_tensors_val,
        seq_lens=patient_seq_lens_val,
        target_labels=target_labels_val,
        init_adj_mat=init_adj_mat,
    )

    patient_tensors_test, patient_seq_lens_test, target_labels_test = df_to_patient_tensors(
        df_test, feature_cols=feature_cols, target_col=target_col, scaler=scaler
    )
    patient_tensors_test = pad_sequence(patient_tensors_test, batch_first=True)
    init_adj_mat = dag_fully_connected(num_nodes=len(feature_cols), add_self_loops=False)

    dataset_test = GraphClassificationDataset(
        data=patient_tensors_test,
        seq_lens=patient_seq_lens_test,
        target_labels=target_labels_test,
        init_adj_mat=init_adj_mat,
    )

    return dataset_train, dataset_val, dataset_test
