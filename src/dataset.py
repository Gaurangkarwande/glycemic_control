from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from src.constants import INPUT_COVARIATES


SCALER_TYPE = RobustScaler


class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.

    """

    def __init__(
        self,
        data: List[List[torch.tensor]],
        labels: List[List[torch.tensor]],
        indices: List[List[Tuple[int, int]]],
        num_samples: int,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> None:

        """
        Args:
            data: the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.
            labels : the entire training labels
            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.
            num_samples: total number of input output pairs
            enc_seq_len: the desired length of the input sequence given to the
                     the first layer of the transformer model.
            target_seq_len: the desired length of the target sequence (the output of the model)
            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()

        self.indices = self.stack_indices(labels, indices)
        assert len(self.indices) == num_samples

        self.data = torch.vstack(data)

        self.labels = torch.vstack(labels)

        self.enc_seq_len = enc_seq_len

        self.target_seq_len = target_seq_len

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """

        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]
        labels = self.labels[start_idx:end_idx]

        # print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            data_sequence=sequence,
            label_sequence=labels,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len,
        )

        return src, trg, trg_y

    def get_src_trg(
        self,
        data_sequence: torch.Tensor,
        label_sequence: torch.Tensor,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.
        Args:
            data_sequence: a tensor of shape (n, NUM_COVARIATES) where
                    n = encoder input length + target sequence length
            label_sequence: a tensor of shpae (n, 1) where
                    n = encoder input length + target sequence length
            enc_seq_len: the desired length of the input to the transformer encoder
            target_seq_len: the desired length of the target sequence (the
                            one against which the model output is compared)
        Return:
            src: used as input to the transformer encoder (enc_seq_len, NUM_COVARIATES)
            trg: used as input to the transformer decoder (target_seq_len, 1)
            trg_y: the target sequence against which the model output
                is compared when computing loss. (target_seq_len, 1)

        """
        assert (
            len(data_sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"

        # encoder input
        src = data_sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = data_sequence[enc_seq_len - 1 : len(label_sequence) - 1]

        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = label_sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return (
            src,
            trg,
            trg_y.squeeze(-1),
        )

    def stack_indices(
        self, labels: List[List[torch.Tensor]], indices: List[List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        Stacks the patient specific indices list into one cohesive list that can iterated upon
        in the dataloader

        Args:
            labels : the output labels for each input observation
            indices : the list of patient specific indices

        Returns: stacked indices with mapping to patient specific indices
        """

        stacked_indices = []
        prev_cum_len = 0
        for labels, patient_indices in zip(labels, indices):
            for endpoints in patient_indices:
                start, end = endpoints
                stacked_indices.append((prev_cum_len + start, prev_cum_len + end))
            prev_cum_len += len(labels)
        return stacked_indices


class GraphDataset(Dataset):
    """
    Dataset class used for transformer models.

    """

    def __init__(
        self,
        data: List[List[torch.tensor]],
        labels: List[List[torch.tensor]],
        indices: List[List[Tuple[int, int]]],
        num_samples: int,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> None:

        """
        Args:
            data: the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.
            labels : the entire training labels
            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.
            num_samples: total number of input output pairs
            enc_seq_len: the desired length of the input sequence given to the
                     the first layer of the transformer model.
            target_seq_len: the desired length of the target sequence (the output of the model)
            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()

        self.indices = self.stack_indices(labels, indices)
        assert len(self.indices) == num_samples

        self.data = torch.vstack(data)

        self.labels = torch.vstack(labels)

        self.enc_seq_len = enc_seq_len

        self.target_seq_len = target_seq_len

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """

        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]
        labels = self.labels[start_idx:end_idx]

        # print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg_y = self.get_src_trg(
            data_sequence=sequence,
            label_sequence=labels,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len,
        )

        return src, trg_y

    def get_src_trg(
        self,
        data_sequence: torch.Tensor,
        label_sequence: torch.Tensor,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.
        Args:
            data_sequence: a tensor of shape (n, NUM_COVARIATES) where
                    n = encoder input length + target sequence length
            label_sequence: a tensor of shpae (n, 1) where
                    n = encoder input length + target sequence length
            enc_seq_len: the desired length of the input to the transformer encoder
            target_seq_len: the desired length of the target sequence (the
                            one against which the model output is compared)
        Return:
            src: used as input to the transformer encoder (enc_seq_len, NUM_COVARIATES)
            trg: used as input to the transformer decoder (target_seq_len, 1)
            trg_y: the target sequence against which the model output
                is compared when computing loss. (target_seq_len, 1)

        """
        assert (
            len(data_sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"

        # encoder input
        src = data_sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all

        # The target sequence against which the model output will be compared to compute loss
        trg_y = label_sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return (
            src,
            trg_y.squeeze(-1),
        )

    def stack_indices(
        self, labels: List[List[torch.Tensor]], indices: List[List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """
        Stacks the patient specific indices list into one cohesive list that can iterated upon
        in the dataloader

        Args:
            labels : the output labels for each input observation
            indices : the list of patient specific indices

        Returns: stacked indices with mapping to patient specific indices
        """

        stacked_indices = []
        prev_cum_len = 0
        for labels, patient_indices in zip(labels, indices):
            for endpoints in patient_indices:
                start, end = endpoints
                stacked_indices.append((prev_cum_len + start, prev_cum_len + end))
            prev_cum_len += len(labels)
        return stacked_indices


def stack_dataset_featues_target(
    dataset: GraphDataset,
) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    """Stack the features and targets from the created dataset

    Args:
        dataset: an iterable temporal dataset

    Returns: the stacked features and targets
    """

    features = []
    targets = []
    for snapshot in dataset:
        features.append(snapshot[0].T.numpy())
        targets.append(snapshot[1].numpy())
    return features, targets


def get_adjacency_matrix(df_dag: Optional[pd.DataFrame]) -> np.ndarray:
    """Get adjacency matrix from dataframe

    Args:
        df_dag: the dag dataframe

    Returns: the adjacency matrix
    """

    if df_dag is None:
        return None
    adj_matrix = df_dag[INPUT_COVARIATES].to_numpy()  # this is an upper triangular matrix
    for row in range(adj_matrix.shape[0]):
        for col in range(row):
            adj_matrix[row, col] = adj_matrix[col, row]
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
    scaler_x: SCALER_TYPE,
    scaler_y: SCALER_TYPE,
) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    """Convert the dataframe to tensor data

    Args :
        df : the data in dataframe form
        feature_cols : the list of covariates we want to use as features in the multivariate
            timeseries
        target_col : the name of the column having glucose values
        scaler : the preprocessing scaler used to normalize the data

    Returns : each patient timeseries as (feature_tensor, target_tensor)
    """

    patient_features_list = []
    patient_target_list = []
    df_patient_group = df.groupby(by="subject_id")
    for _, df_patient in df_patient_group:
        features = df_patient[feature_cols].to_numpy()
        if scaler_x is not None:
            features = scaler_x.transform(features)

        target = df_patient[target_col].to_numpy()
        if scaler_y is not None:
            target = scaler_y.transform(target)

        patient_features_list.append(torch.tensor(features, dtype=torch.float32))
        patient_target_list.append(torch.tensor(target, dtype=torch.float32))

    return patient_features_list, patient_target_list
