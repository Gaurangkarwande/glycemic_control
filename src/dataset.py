from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.

    """

    def __init__(
        self,
        data: torch.tensor,
        indices: list,
        enc_seq_len: int,
        target_seq_len: int,
    ) -> None:

        """
        Args:
            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.
            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.
            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.
            target_seq_len: int, the desired length of the target sequence (the output of the model)
            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()

        self.indices = indices

        self.data = data

        print("From get_src_trg: data size = {}".format(data.size()))

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

        # print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len,
        )

        return src, trg, trg_y

    def get_src_trg(
        self, sequence: torch.Tensor, enc_seq_len: int, target_seq_len: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.
        Args:
            sequence: tensor, a 1D tensor of length n where
                    n = encoder input length + target sequence length
            enc_seq_len: int, the desired length of the input to the transformer encoder
            target_seq_len: int, the desired length of the target sequence (the
                            one against which the model output is compared)
        Return:
            src: tensor, 1D, used as input to the transformer encoder
            trg: tensor, 1D, used as input to the transformer decoder
            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss.

        """
        assert (
            len(sequence) == enc_seq_len + target_seq_len
        ), "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len - 1 : len(sequence) - 1]  # noqa E203

        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return (
            src,
            trg,
            trg_y.squeeze(-1),
        )  # change size from [batch_size, target_seq_len, num_features] to
        # [batch_size, target_seq_len]


def get_normalizing_scaler(
    X_train: np.array, scaler_type: str = "minmax"
) -> Union[MinMaxScaler, StandardScaler]:
    """Get the preprocessing scaler that normalizes the feature values

    Args:
        X_train : the training dataset

    Returns: the fitted scaler. Could be one of MinMaxScaler or StandardScaler
    """
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    scaler.fit(X_train)
    return scaler


def df_to_patient_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    scaler: Union[MinMaxScaler, StandardScaler],
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
        features = scaler.transform(df_patient[feature_cols].to_numpy())
        target = df_patient[target_col].to_numpy()

        patient_features_list.append(torch.tensor(features))
        patient_target_list.append(torch.tensor(target))

    return patient_features_list, patient_target_list
