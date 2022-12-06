import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.constants import INPUT_COVARIATES, TARGET_COL
from src.dataloader import get_dataloaders
from src.models.temporal_gnn import RecurrentGCN_classification
from src.utils import EarlyStopping, LRScheduler, get_timestamp
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

GLOBAL_SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(
    model: torch.nn.Module,
    dataloader: StaticGraphTemporalSignal,
    optimizer: Any,
    criterion: Callable,
) -> Tuple[float, float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        optimizer: the optimizer to train the model
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        accuracy: the classification accuracy
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = correct = total = 0
    model.train()
    for batch_id, snapshot in enumerate(dataloader):
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_attr = snapshot.edge_attr.to(device)
        y = snapshot.y.to(device)
        batch = snapshot.batch.to(device)
        optimizer.zero_grad()

        # find predictions
        pred = model(x, edge_index, edge_attr, batch)
        loss = criterion(pred, y)

        # backpropagate
        loss.backward()
        optimizer.step()

        # get metrics
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        epoch_loss += float(loss.item())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch


def evaluate(
    model: torch.nn.Module, dataloader: StaticGraphTemporalSignal, criterion: Callable
) -> Tuple[float, float, float]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        accuracy: the classification accuracy
        time: the time taken for a single epoch
    """

    start = time.time()
    epoch_loss = correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch_id, snapshot in enumerate(dataloader):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = snapshot.y.to(device)
            batch = snapshot.batch.to(device)

            # find predictions
            pred = model(x, edge_index, edge_attr, batch)
            loss = criterion(pred, y)

            # get metrics
            _, pred_classes = pred.max(1)
            total += y.size(0)
            correct += float(pred_classes.eq(y).sum().item())
            epoch_loss += float(loss.item())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch


def inference(
    model: torch.nn.Module, dataloader: StaticGraphTemporalSignal, criterion: Callable
) -> Tuple[float, float, float, List[int], List[int]]:
    """Train the model

    Args:
        model: the model to train
        dataloader: the training dataloader
        criterion: the loss criterion

    Returns:
        avg_loss: the average loss for the epoch
        time: the time taken for inference
        predictions: the predicted glucose classes
        gts: the ground truth glucose classes
    """

    start = time.time()
    epoch_loss = correct = total = 0
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

            # find predictions
            pred = model(x, edge_index, edge_attr, batch)
            loss = criterion(pred, y)

            # get metrics
            _, pred_classes = pred.max(1)
            total += y.size(0)
            correct += float(pred_classes.eq(y).sum().item())
            epoch_loss += float(loss.item())

            gts.extend(y.tolist())
            predictions.extend(pred.tolist())

    # aggregate metrics for epoch
    accuracy = correct / total
    avg_loss = epoch_loss / len(dataloader)
    time_for_epoch = time.time() - start
    return avg_loss, accuracy, time_for_epoch, predictions, gts


def train_gnn(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    df_dag: Optional[pd.DataFrame],
    dirpath_out: Path,
    logger: logging.Logger,
    verbose: bool = True,
) -> float:
    """Train the transformer model

    Args:
        df_train: training data
        df_test: training data
        df_test: training data
        df_dag: the DAG adjacency matrix
        dirpath_out: path to where training runs will be recorded
        verbose: whether to print details

    Returns: the inference loss
        # TODO: df_results: the dataframe housing the results
    """

    start = time.time()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    # Hyperparams
    batch_size = 128
    lr = 1e-2
    num_epochs = 100

    # Params
    enc_seq_len = 7  # length of input given to encoder
    output_sequence_length = 1  # how many future glucose values to predict
    step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
    num_classes = 5  # number of discrete bins glucose values are divided into

    # Define input variables
    input_variables = INPUT_COVARIATES + TARGET_COL

    logger.info(
        f"Time series params: \nInput sequence lenght: {enc_seq_len} \nOutput sequence lenght:"
        f" {output_sequence_length} \nStep size: {step_size}"
    )

    if verbose:
        print(
            f"Time series params: \nInput sequence lenght: {enc_seq_len} \nOutput sequence lenght:"
            f" {output_sequence_length} \nStep size: {step_size}"
        )

    # create dataset

    dataloader_train, dataloader_valid, dataloader_test, _ = get_dataloaders(
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
        logger=logger,
        verbose=verbose,
        normalize_target=False,
    )

    # create model

    model = RecurrentGCN_classification(node_features=enc_seq_len, num_classes=num_classes)

    # Criterion optimizer early stopping lr_scheduler

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=20, verbose=verbose)
    lr_scheduler = LRScheduler(optimizer, verbose=verbose)

    # transfer to GPU

    train_on_gpu = torch.cuda.is_available()

    if verbose:
        if train_on_gpu:
            print("Training on GPU")
        else:
            print("No GPU available, training on CPU")

    # setup paths

    fpath_loss_curve_plot = dirpath_out.joinpath("loss_curve.jpg")
    fpath_acc_curve_plot = dirpath_out.joinpath("acc_curve.jpg")
    fpath_inference_df = dirpath_out.joinpath("inference.csv")
    dirpath_checkpoint = dirpath_out.joinpath("model_checkpoint")
    dirpath_checkpoint.mkdir()
    fpath_checkpoint = None

    # training
    logger.info("****** Training ******")
    if verbose:
        print("****** Training ******")

    model.to(device)

    train_history_loss = []
    train_history_acc = []

    val_history_loss = []
    val_history_acc = []

    best_valid_acc = 0
    for epoch in range(num_epochs):
        loss_train, acc_train, time_train = train(
            model=model, dataloader=dataloader_train, optimizer=optimizer, criterion=criterion
        )
        loss_valid, acc_valid, _ = evaluate(
            model=model, dataloader=dataloader_valid, criterion=criterion
        )
        if verbose:
            print(
                f"Epoch {epoch}: Train Loss= {loss_train:.3f}, Train Acc= {acc_train:.3f} \t"
                f"Valid Loss= {loss_valid:.3f}, Valid Acc={acc_valid:.3f} \t"
                f" Training Time ={time_train:.2f} s"
            )
        logger.info(
            f"Epoch {epoch}: Train Loss= {loss_train:.3f}, Train Acc= {acc_train:.3f} \t"
            f"Valid Loss= {loss_valid:.3f}, Valid Acc={acc_valid:.3f} \t"
            f" Training Time ={time_train:.2f} s"
        )
        if acc_valid > best_valid_acc:
            best_valid_acc = acc_valid
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_valid_acc,
            }
            fpath_checkpoint = dirpath_checkpoint.joinpath("best_checkpoint.pth")
            torch.save(checkpoint, fpath_checkpoint)
            if verbose:
                print(f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}")
            logger.info(
                f"Checkpoint saved at Epoch : {epoch}, at location : {str(fpath_checkpoint)}"
            )
        lr_scheduler(loss_valid)
        early_stopping(loss_valid)

        train_history_loss.append(loss_train)
        train_history_acc.append(acc_train)

        val_history_loss.append(loss_valid)
        val_history_acc.append(acc_valid)

        if early_stopping.early_stop:
            break

    logger.info(f"Final scheduler state {lr_scheduler.get_final_lr()}\n")
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # save loss curves
    plt.plot(range(len(train_history_loss)), train_history_loss, label="Train CE Loss")
    plt.plot(range(len(val_history_loss)), val_history_loss, label="Val CE Loss")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("CE Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig(fpath_loss_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

    # save loss curves
    plt.plot(range(len(train_history_acc)), train_history_acc, label="Train Accuracy")
    plt.plot(range(len(val_history_acc)), val_history_acc, label="Val Accuracy")
    # plt.ylim(top=5000)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")
    plt.savefig(fpath_acc_curve_plot, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info("******* Finished Training *******")
    if verbose:
        print("******* Finished training ********")

    # inference on test set
    checkpoint = torch.load(fpath_checkpoint)
    model = RecurrentGCN_classification(node_features=enc_seq_len, num_classes=num_classes)
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    loss_test, acc_test, _, predicted_values, gts = inference(
        model=model, dataloader=dataloader_test, criterion=criterion
    )

    # inv_scaled_gt = scaler_y.inverse_transform(gts)
    # inv_scaled_predictions = scaler_y.inverse_transform(predicted_values)
    # inference_result = pd.DataFrame(
    #     zip(
    #         gts.squeeze(),
    #         predicted_values.squeeze(),
    #         inv_scaled_gt.squeeze(),
    #         inv_scaled_predictions.squeeze(),
    #     ),
    #     columns=["gt_scaled", "predicted_scaled", "gt_inv_scaled", "predicted_inv_scaled"],
    # )

    inference_result = pd.DataFrame(zip(gts, predicted_values), columns=["gt", "predicted"])
    inference_result.to_csv(fpath_inference_df)

    time_taken = time.time() - start

    logger.info(f"Final test CE loss: {loss_test}")
    logger.info(f"Final test accuracy: {acc_test}")
    logger.info(f"Total time taken: {time_taken} s")

    if verbose:
        print(f"Final test CE loss: {loss_test}")
        print(f"Final test accuracy: {acc_test}")
        print(f"Total time taken: {time_taken} s")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return loss_test, acc_test


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
