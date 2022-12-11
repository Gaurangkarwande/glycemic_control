"""Aggregating partial dags

This script aggregates the clinical dags by majority voting and creates a new dag for each
threshold in the range [1, 23]. The resultant dags are saved as csv files in original format within
the provided parent directory
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.dataset import get_adjacency_matrix
from src.utils import aggregate_partial_dags, get_timestamp


def aggregate_and_save(
    adj_matrices: List[np.ndarray],
    majority_threshold: int,
    node_names: List[str],
    dirpath_save: Path,
) -> None:
    """Aggregate the partial dags and save as a csv file

    Args:
        adj_matrices: the list of partial dags to aggregate
        majority_threshold: the threshold value to use for aggregation
        node_names: the list of node names in correct sequence
        dirpath_save: the directory where the output csv will be saved
    """

    fpath_save = dirpath_save.joinpath(f"aggregated_dag_threshold_{majority_threshold}.csv")
    adj_matrix_agg = aggregate_partial_dags(
        partial_dags=adj_matrices, majority_threshold=majority_threshold
    )
    df_adj_matrix_agg = pd.DataFrame(adj_matrix_agg, columns=node_names, index=node_names)
    df_adj_matrix_agg.to_csv(fpath_save)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate the partial dags")
    parser.add_argument(
        "--dirpath_partial_dags",
        type=Path,
        required=True,
        help="the directory path where all the partial dags to aggregate are stored as csv files",
    )
    parser.add_argument(
        "--dirpath_save",
        type=Path,
        required=True,
        help="the dirpath where the set of aggregated dags will be saved",
    )

    args = parser.parse_args()
    return args


def main():
    """The entry point to the scrip"""
    args = parse_args()
    df_dags = []

    for fpath_dag in args.dirpath_partial_dags.iterdir():
        df_dags.append(pd.read_csv(fpath_dag))

    adj_matrices = []
    for df_dag in df_dags:
        adj_matrices.append(get_adjacency_matrix(df_dag))

    node_names = df_dags[1].columns[1:]

    dirpath_save = args.dirpath_save.joinpath(get_timestamp() + args.dirpath_partial_dags.name)
    dirpath_save.mkdir()

    for majority_threshold in range(1, len(adj_matrices)):
        aggregate_and_save(
            adj_matrices=adj_matrices,
            majority_threshold=majority_threshold,
            node_names=node_names,
            dirpath_save=dirpath_save,
        )


if __name__ == "__main__":
    main()
