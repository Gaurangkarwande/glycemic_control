import argparse
from itertools import repeat
from pathlib import Path
from src.train_gnn import train_gnn

import pandas as pd
from typing import Any, Dict, List, Optional
import logging

import torch.multiprocessing as mp

from src.utils import get_timestamp
import json


def my_custom_logger(logger_name, logger_fpath, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    message_format_string = "%(asctime)s %(message)s"
    time_format_string = "%m/%d/%Y %H:%M:%S"
    log_format = logging.Formatter(message_format_string, time_format_string)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_fpath)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def run_experiment(
    fpath_dag: Optional[Path],
    dag_type: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    dirpath_results: Path,
) -> float:
    """Runs the experiment for a single dag

     Args:
        fpath_dag: the path to the dag csv
        dag_type: diabetic, non-diabetic, OR all
        df_train: training data
        df_test: training data
        df_test: training data
        dirpath_results: path to where results will be recorded

    Returns: the inference loss
        # TODO: df_results: the dataframe housing the results
    """

    dag_name = fpath_dag.stem if fpath_dag is not None else "complete"
    df_dag = pd.read_csv(fpath_dag) if fpath_dag is not None else None

    dirpath_out = dirpath_results.joinpath(f"{dag_type}_{dag_name}")
    dirpath_out.mkdir()

    # create logger
    logger = my_custom_logger(
        logger_name=f"graph_{dag_type}_{dag_name}",
        logger_fpath=dirpath_out.joinpath(f"graph_{dag_type}_{dag_name}.log"),
    )

    # log paths

    logger.info(f"fpath_dag: {fpath_dag}")
    logger.info(f"dag_type: {str(dag_type)}")
    logger.info(f"dag_name: {str(dag_name)}")

    print(f"Started training for DAG: {dag_type}_{dag_name}")

    try:
        test_loss = train_gnn(
            df_train=df_train,
            df_valid=df_valid,
            df_test=df_test,
            df_dag=df_dag,
            dirpath_out=dirpath_out,
            logger=logger,
            verbose=False,
        )
    except Exception as e:
        print(f"ERROR for DAG: {dag_type}_{dag_name} --> {e}")
        test_loss = None
    return test_loss


def run_dag_set(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    fpaths_dags: List[Optional[Path]],
    dag_types: List[str],
    dirpath_results: Path,
) -> Dict[str, Any]:
    """Runs experiments for all dags

    Args:
        df_train: training data
        df_test: training data
        df_test: training data
        fpaths_dag: list of dag filepaths
        dag_types: list of dag types. diabetic, non-diabetic, OR all
        dirpath_results: path to where results will be recorded

    Returns: inference results for all dags

    """

    assert len(fpaths_dags) == len(dag_types)

    dag_losses = []
    with mp.Pool(processes=4) as pool:
        for test_loss in pool.starmap(
            run_experiment,
            zip(
                fpaths_dags,
                dag_types,
                repeat(df_train),
                repeat(df_valid),
                repeat(df_test),
                repeat(dirpath_results),
            ),
        ):
            dag_losses.append(test_loss)

    # for fpath_dag, dag_type in zip(fpaths_dags, dag_types):
    #     test_loss = run_experiment(
    #         fpath_dag=fpath_dag,
    #         dag_type=dag_type,
    #         df_train=df_train,
    #         df_valid=df_valid,
    #         df_test=df_test,
    #         dirpath_results=dirpath_results,
    #     )
    #     dag_losses.append(test_loss)

    dag_inference_dict = {}
    for fpath_dag, dag_type, dag_loss in zip(fpaths_dags, dag_types, dag_losses):
        if fpath_dag is not None:
            dag_name = f"{dag_type}_{fpath_dag.stem}"
        else:
            dag_name = f"{dag_type}_complete"
        dag_inference_dict[dag_name] = dag_loss

    return dag_inference_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Parser for training transformer model")
    parser.add_argument(
        "--fpath_train_data", type=Path, help="path to train dataframe", required=True
    )
    parser.add_argument(
        "--fpath_valid_data", type=Path, help="path to validation dataframe", required=True
    )
    parser.add_argument(
        "--fpath_test_data", type=Path, help="path to test dataframe", required=True
    )
    parser.add_argument("--dirpath_dags", type=Path, help="path to dags directory", required=True)
    parser.add_argument("--dag_type", type=str, help="diabetic, non, or all", required=True)
    parser.add_argument(
        "--dirpath_results",
        type=Path,
        help="path to where training runs will be recorded",
        required=True,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # set output dirpath
    dirpath_dag_set_results = args.dirpath_results.joinpath(get_timestamp() + "dag_set")
    dirpath_dag_set_results.mkdir()

    # load data
    df_train = pd.read_csv(args.fpath_train_data)
    df_valid = pd.read_csv(args.fpath_valid_data)
    df_test = pd.read_csv(args.fpath_test_data)

    # setup dag info

    fpaths_dags = list(args.dirpath_dags.iterdir())
    fpaths_dags.append(None)  # add fully connected dag
    dag_types = [args.dag_type] * len(fpaths_dags)

    # run experiment
    mp.set_start_method("spawn")

    inference_dict = run_dag_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        fpaths_dags=fpaths_dags[-8:],
        dag_types=dag_types[-8:],
        dirpath_results=dirpath_dag_set_results,
    )

    # save inference results

    with open(dirpath_dag_set_results.joinpath("inference.json"), "w") as fp:
        json.dump(inference_dict, fp, indent=4)
