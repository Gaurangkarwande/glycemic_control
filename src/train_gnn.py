import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from src.constants import INPUT_COVARIATES, TARGET_COL
from src.dataloader import get_dataloaders
from src.models.temporal_gnn import RecurrentGCN
from src.utils import (
    EarlyStopping,
    LRScheduler,
    get_timestamp,
)

GLOBAL_SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(
    model: torch.nn.Module,
    dataloader: StaticGraphTemporalSignal,
    optimizer: Any,
) -> Tuple[float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        optimizer: the optimizer to train the model

    Returns:
        avg_loss: the average loss for the epoch
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = 0
    model.train()
    for batch_id, snapshot in enumerate(dataloader):
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        y = snapshot.y.to(device)
        batch = snapshot.batch.to(device)
        y_hat = model(x, edge_index, edge_attr, batch)
        loss = torch.mean((y_hat - y) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    time_for_epoch = time.time() - start
    return epoch_loss / len(dataloader), time_for_epoch


def evaluate(
    model: torch.nn.Module,
    dataloader: StaticGraphTemporalSignal,
) -> Tuple[float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader

    Returns:
        avg_loss: the average loss for the epoch
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_id, snapshot in enumerate(dataloader):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)
            batch = snapshot.batch.to(device)
            y_hat = model(x, edge_index, edge_attr, batch)
            loss = torch.mean((y_hat - y) ** 2)
            epoch_loss += loss.item()
    time_for_epoch = time.time() - start
    return epoch_loss / len(dataloader), time_for_epoch


def inference(
    model: torch.nn.Module,
    dataloader: StaticGraphTemporalSignal,
) -> Tuple[float, float, List[float], List[float]]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader

    Returns:
        avg_loss: the average loss for the epoch
        time: the time taken for inference
        predictions: the predicted glucose values. These are normalized
        gts: the ground truth values. These are normalized.
    """

    start = time.time()
    epoch_loss = 0
    model.eval()
    gts = []
    predictions = []
    with torch.no_grad():
        for batch_id, snapshot in enumerate(dataloader):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)
            batch = snapshot.batch.to(device)
            y_hat = model(x, edge_index, edge_attr, batch)
            loss = torch.mean((y_hat - y) ** 2)
            epoch_loss += loss.item()
            gts.extend(y.tolist())
            predictions.extend(y_hat.tolist())
    time_taken = time.time() - start
    return (
        epoch_loss / len(dataloader),
        time_taken,
        np.array(predictions).reshape(-1, 1),
        np.array(gts).reshape(-1, 1),
    )


def train_gnn(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    df_dag: Optional[pd.DataFrame],
    dirpath_out: Path,
    logger: logging.Logger,
) -> None:
    """Train the transformer model

    Args:
        df_train: training data
        df_test: training data
        df_test: training data
        df_dag: the DAG adjacency matrix
        dirpath_out: path to where training runs will be recorded

    Returns:
        # TODO: df_results: the dataframe housing the results
    """

    start = time.time()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # Hyperparams
    batch_size = 256
    lr = 1e-2
    num_epochs = 100

    # Params
    enc_seq_len = 6  # length of input given to encoder
    output_sequence_length = 1  # how many future glucose values to predict
    step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step

    # Define input variables
    input_variables = INPUT_COVARIATES + TARGET_COL

    logger.info(
        f"Time series params: \nInput sequence lenght: {enc_seq_len} \nOutput sequence lenght:"
        f" {output_sequence_length} \nStep size: {step_size}"
    )
    print(
        f"Time series params: \nInput sequence lenght: {enc_seq_len} \nOutput sequence lenght:"
        f" {output_sequence_length} \nStep size: {step_size}"
    )

    # logger.info(f"Model hyperparameters: \nBatch size: {batch_size} \nLearning rate: {lr}")
    # print(f"Model hyperparameters: \nBatch size: {batch_size} \nLearning rate: {lr}")

    # create dataset

    dataloader_train, dataloader_valid, dataloader_test, scaler_y = get_dataloaders(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        df_dag=df_dag,
        input_variables=input_variables,
        target_variable=TARGET_COL,
        enc_seq_len=enc_seq_len,
        output_sequence_length=output_sequence_length,
        step_size=step_size,
        batch_size=batch_size,
        logger=logger
    )

    # create model

    model = RecurrentGCN(node_features=enc_seq_len, batch_size=batch_size)

    # Criterion optimizer early stopping lr_scheduler

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=20)
    lr_scheduler = LRScheduler(optimizer)

    # transfer to GPU

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("Training on GPU")
    else:
        print("No GPU available, training on CPU")

    # setup paths

    fpath_curve_plot = dirpath_out.joinpath("loss_curve.jpg")
    fpath_inference_df = dirpath_out.joinpath("inference.csv")
    dirpath_checkpoint = dirpath_out.joinpath("model_checkpoint")
    dirpath_checkpoint.mkdir()
    fpath_checkpoint = None

    # training
    logger.info("****** Training ******")
    print("****** Training ******")
    model.to(device)

    train_history_loss = []
    val_history_loss = []
    best_valid_rmse = float("inf")
    for epoch in range(num_epochs):
        loss_train, time_train = train(
            model=model, dataloader=dataloader_train, optimizer=optimizer
        )
        loss_valid, _ = evaluate(model=model, dataloader=dataloader_valid)
        print(
            f"Epoch {epoch}: Train Loss= {loss_train:.3f} \t Valid Loss= {loss_valid:.3f}, \t"
            f" Training Time ={time_train:.2f} s"
        )
        logger.info(
            f"Epoch {epoch}: Train Loss= {loss_train:.3f} \t Valid Loss= {loss_valid:.3f}, \t"
            f" Training Time ={time_train:.2f} s"
        )
        if loss_valid < best_valid_rmse:
            best_valid_rmse = loss_valid
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_rmse": best_valid_rmse,
            }
            fpath_checkpoint = dirpath_checkpoint.joinpath("best_checkpoint.pth")
            torch.save(checkpoint, fpath_checkpoint)
            print(f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}")
            logger.info(
                f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}"
            )
        lr_scheduler(loss_valid)
        early_stopping(loss_valid)

        train_history_loss.append(loss_train)
        val_history_loss.append(loss_valid)

        if early_stopping.early_stop:
            break

    logger.info(f"Final scheduler state {lr_scheduler.get_final_lr()}\n")
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # save curves
    plt.plot(range(len(train_history_loss)), train_history_loss, label="Train MSE")
    plt.plot(range(len(val_history_loss)), val_history_loss, label="Val MSE")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(fpath_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info("******* Finished Training *******")
    print("******* Finished training ********")

    # inference on test set
    checkpoint = torch.load(fpath_checkpoint)
    model = RecurrentGCN(node_features=enc_seq_len, batch_size=batch_size)
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    loss_test, _, predicted_values, gts = inference(model=model, dataloader=dataloader_test)
    inv_scaled_gt = scaler_y.inverse_transform(gts)
    inv_scaled_predictions = scaler_y.inverse_transform(predicted_values)
    inference_result = pd.DataFrame(
        zip(
            gts.squeeze(),
            predicted_values.squeeze(),
            inv_scaled_gt.squeeze(),
            inv_scaled_predictions.squeeze(),
        ),
        columns=["gt_scaled", "predicted_scaled", "gt_inv_scaled", "predicted_inv_scaled"],
    )
    inference_result.to_csv(fpath_inference_df)

    time_taken = time.time() - start

    logger.info(f"Final test MSE loss: {loss_test}")
    print(f"Final test MSE loss: {loss_test}")
    print(f"Total time taken: {time_taken} s")
    logger.info(f"Total time taken: {time_taken} s")


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
    parser.add_argument(
        "--fpath_dag", type=Path, help="path to test dataframe", required=False, default=None
    )
    parser.add_argument(
        "--dag_type", type=str, help="diabetic, non, or all", required=False, default=None
    )
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
    dirpath_save = args.dirpath_results.joinpath(get_timestamp() + "training_record")
    dirpath_save.mkdir()
    df_train = pd.read_csv(args.fpath_train_data)
    df_valid = pd.read_csv(args.fpath_valid_data)
    df_test = pd.read_csv(args.fpath_test_data)
    df_dag = pd.read_csv(args.fpath_dag) if args.fpath_dag is not None else None
    dag_type = args.dag_type if args.dag_type is not None else ""
    dag_name = args.fpath_dag.stem if args.fpath_dag is not None else "fully_connected"

    # create logger
    logging.basicConfig(
        filename=dirpath_save.joinpath(f"graph_{dag_type}_{dag_name}.log"),
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("graph")

    # log paths

    logger.info(f"fpath_train: {str(args.fpath_train_data)}")
    logger.info(f"fpath_train: {str(args.fpath_valid_data)}")
    logger.info(f"fpath_train: {str(args.fpath_test_data)}")
    logger.info(f"fpath_dag: {args.fpath_dag}")
    logger.info(f"dag_type: {str(dag_type)}")
    logger.info(f"dag_name: {str(dag_name)}")

    train_gnn(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        df_dag=df_dag,
        dirpath_out=dirpath_save,
        logger=logger,
    )
