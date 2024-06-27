# file/system manipulation
import os
import sys
import argparse
from typing import Any, Tuple, Union, Dict

# logging utilities
import logging
import wandb
from time import time
from tqdm import tqdm

# data manipulation
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
import pandas as pd

# modeling
import torchmetrics
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# custom
dataset = __import__("3_dataset")
model_file = __import__("4_model")


def parse_cli() -> argparse.Namespace:
    """command line interface parser to simplify model training kickoff

    Returns:
        argparse.ArgumentParser().parse_args: trigger argument parsing
    """
    # specify parser arguments and thier descriptions + defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--parquet-path-train",
        type=str,
        default="",
        help="path to directory of train data assets",
    )
    parser.add_argument(
        "--parquet-path-val",
        type=str,
        default="",
        help="path to directory of validation data assets",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            "passing_intention_model_training", str(time()).split(".")[0] + "/"
        ),
        help="default directory to save training outputs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of CPU threads for data loading",
    )
    parser.add_argument(
        "--is-training",
        type=bool,
        default=False,
        help="True if running not inference, else False",
    )

    return parser.parse_args()


def create_data_loader(
    parquet_path: str,
    row_dim: int,
    batch_size: int,
    is_training: bool,
    num_workers: int,
) -> DataLoader[Any]:
    """create data loader for a PyTorch Dataset

    Args:
        data_dir (str): file path to dataset
        batch_size (int): size of training batch (dictated my system memory)
        is_training (bool): boolean that dictates wether or not to shuffle dataset
        num_workers (int): number of threads for data loading

    Returns:
        torch.utils.data.DataLoader: PyTorch DataLoader that serves up data for training
    """
    # if evaluating, order matters, otherwise shuffle such that order does not influence learning
    shuffle = is_training

    ds = dataset.PassingIntentionDataset(parquet_path=parquet_path, row_dim=row_dim)

    # create PyTorch dataloader where last incomplete batch is dropped
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def save_stats(out_dir: str, value: float, out_file: str) -> None:
    """manual logging of values for posterity

    Args:
        out_dir (str): out data directory to save manual log files to
        value (float): value to write
        out_file (str): specific filename to write to
    """
    # opens file in append mode and writes logging-data-of-interest
    with open(os.path.join(out_dir, out_file), "a") as out_write:
        out_write.write(str(value) + "\n")


def train_epoch(
    train_data_loader: DataLoader[Any],
    model: nn.Module,
    loss_func: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    epoch: int,
    tensorboard_writer: SummaryWriter,
    device: str,
) -> npt.NDArray[np.float32]:
    """implements within-epoch training step

    Args:
        train_data_loader (torch.utils.data.DataLoader): PyTorch DataLoader for serving up training feature-label pairs
        model (torch.nn.Module): model for training
        loss_func (torch.nn.CrossEntropyLoss): torch loss function for dictating loss landscape and location within it
        optimizer (torch.optim.Adam): torch optimizer for dictating step size and direction in loss landscape
        epoch (int): epoch number for logging
        device (str): cuda, mps, or cpu

    Returns:
        np.ndarray: numpy array of training loss for logging (0-dim array with scalar)
    """
    # switch to traning mode
    model.train()
    loss_ls = []
    # load feature-label pairs into memory
    for X, y, l in tqdm(train_data_loader, colour="green"):
        # send to GPU
        # X = pack_padded_sequence(X, lengths=l, batch_first=True, enforce_sorted=False)
        X, y = X.to(device=device), y.to(device=device)
        # reset gradients
        optimizer.zero_grad()
        # perform forward pass
        logits = model(X)
        # evaluate loss + compute gradients
        loss = loss_func(logits, y)
        # backpropogate
        loss.backward()
        # step in computed direction + step size in loss landscape
        optimizer.step()
        # save loss for logging
        loss = loss.mean()
        loss_ls.append(loss.item())
    agg_loss = np.vstack(loss_ls).mean()
    tensorboard_writer.add_scalar("Loss/train", agg_loss, epoch)
    return agg_loss


@torch.no_grad()
def eval_func(
    val_data_loader: DataLoader[Any],
    model: nn.Module,
    loss_func: nn.CrossEntropyLoss,
    metric_func: torchmetrics.F1Score,
    epoch: int,
    tensorboard_writer: SummaryWriter, 
    device: str,
) -> Tuple[Union[int, float], Any]:
    """implements within-epoch evaluation

    Args:
        val_data_loader (torch.utils.data.DataLoader): PyTorch DataLoader for serving up validation feature-label pairs
        model (torchvision.models.resnet18): ResNet-18 model for training
        loss_func (torch.nn.CrossEntropyLoss): torch loss function for dictating loss landscape and location within it
        metric_func (torchmetrics.F1Score): metric function to judge training task performance
        epoch (int): epoch number for logging
        device (str): cuda, mps, or cpu

    Returns:
        torch.Tensor: torch tensor of evaluation loss for logging (0-dim tensor with scalar)
        torch.Tensor: torch tensor of task performance metric for logging (0-dim array with scalar)
    """
    # switch to evaluation mode
    model.eval()
    preds = []
    gt = []
    loss_ls = []
    # load feature-label pairs into memory
    for X, y, l in tqdm(val_data_loader, colour="red"):
        # send to GPU
        # X = pack_padded_sequence(X, lengths=l, batch_first=True, enforce_sorted=False)
        X, y = X.to(device=device), y.to(device=device)
        # get model guess
        logits = model(X)
        # post-process guess
        _, preds_ = torch.max(logits, 1)
        _, gt_ = torch.max(y, 1)
        # compute evaluation loss to compare with training loss
        loss = loss_func(logits, y)
        # log loss value
        # save guesses and loss values
        loss_ls.append(loss)
        preds.append(preds_.type(torch.int16).cpu())
        gt.append(gt_.type(torch.int16).cpu())
    # compute task performace metric
    metric = metric_func(torch.cat(preds), torch.cat(gt))
    # log task performance metric
    agg_loss = torch.vstack(loss_ls).mean().item()
    tensorboard_writer.add_scalar("Loss/val", agg_loss, epoch)
    tensorboard_writer.add_scalar("F1/val", metric, epoch)
    return agg_loss, metric


def train_loop(
    train_data_loader: DataLoader[Any],
    val_data_loader: DataLoader[Any],
    epochs: int,
    model: nn.Module,
    loss_func: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    out_dir: str,
    scheduler: optim.lr_scheduler.StepLR,
    metric_func: torchmetrics.F1Score,
    device: str,
) -> None:
    """model training driver function

    Args:
        train_data_loader (torch.utils.data.DataLoader): PyTorch DataLoader for serving up training feature-label pairs
        val_data_loader (torch.utils.data.DataLoader): PyTorch DataLoader for serving up validation feature-label pairs
        epochs (int): number of epochs to train for
        model (torchvision.models.resnet18): ResNet-18 model for training
        loss_func (torch.nn.CrossEntropyLoss): torch loss function for dictating loss landscape and location within it
        optimizer (torch.optim.Adam): torch optimizer for dictating step size and direction in loss landscape
        out_dir (str): file path to out directory to save model checkpoints and manual logs to
        scheduler (optim.lr_scheduler.StepLR): leanring rate scheduler to increment/decrement learning rate
        metric_func (torchmetrics.F1Score): metric function to judge training task performance
        device (str): cuda, mps, or cpu
    """
    # C-style find minimum initalization
    best_loss = np.float32(sys.maxsize)
    tensorboard_writer = SummaryWriter()
    # epoch training + evaluation
    for epoch in tqdm(range(epochs), colour="orange"):
        start_time = time()
        # execute epoch training
        train_loss = train_epoch(
            train_data_loader,
            model,
            loss_func,
            optimizer,
            epoch,
            tensorboard_writer,
            device,
        )
        # execute epoch evaluation
        eval_loss, eval_metric = eval_func(
            val_data_loader,
            model,
            loss_func,
            metric_func,
            epoch,
            tensorboard_writer,
            device,
        )

        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            # checkpoint = None
        torch.save(
            model.state_dict(), os.path.join(out_dir, "model.pth")
        )
        checkpoint = Checkpoint.from_directory(out_dir)

        # Send the current training result back to Tune
        # step learning rate scheduler
        scheduler.step()

        # save model checkpoint if evaluation loss improves
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_dict = {
                "epoch": epochs,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_metric": eval_metric.item(),
                "model_state_dict": model.state_dict(),
            }
            torch.save(best_dict, os.path.join(out_dir, "best.pt"))

        # log some information to the console
        print(
            "Epoch #",
            epoch,
            ":",
            "Train loss:",
            train_loss,
            "Val loss:",
            eval_loss,
            "Val metric:",
            eval_metric.item(),
        )
        print("Epoch Time:", str(time() - start_time))

        # manually save some information for posterity
        save_stats(out_dir, float(train_loss), "train_loss.txt")
        save_stats(out_dir, float(eval_loss), "eval_loss.txt")
        save_stats(out_dir, float(eval_metric.item()), "eval_metric.txt")

        train.report(
            {
                "eval_metric": float(eval_metric.item()),
                "eval_loss": float(eval_loss),
                "train_loss": float(train_loss),
            },
            checkpoint=checkpoint,
        )


def training_wrapper(search_space):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row_dim = max(
        pd.read_parquet(args.parquet_path_val).groupby("obj_index").size().max(),
        pd.read_parquet(args.parquet_path_train).groupby("obj_index").size().max(),
    )

    # create DataLoaders
    train_data_loader = create_data_loader(
        parquet_path=args.parquet_path_train,
        row_dim=row_dim,
        batch_size=search_space["batch_size"],
        is_training=args.is_training,
        num_workers=args.num_workers,
    )

    val_data_loader = create_data_loader(
        parquet_path=args.parquet_path_val,
        row_dim=row_dim,
        batch_size=search_space["batch_size"],
        is_training=args.is_training,
        num_workers=args.num_workers,
    )

    # log size to console
    print("TRAIN DATALOADER LENGTH:", len(train_data_loader))
    print("VAL DATALOADER LENGTH:", len(val_data_loader))

    model = model_file.TimeSeriesClassifier(
        num_features=6,
        num_classes=2,
        hidden_size=search_space["hidden_size"],
        num_layers=search_space["num_layers"],
        dropout_fraction=search_space["dropout"],
    )

    # send model to GPU
    model.to(device=device)

    # log model parameters to train to console
    print("MODEL SIZE:", sum(p.numel() for p in model.parameters()), "parameters")

    # initalize loss function, optimizer, learning-rate scheduler, and task metric
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=search_space["learning_rate"],
        weight_decay=search_space["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
    metric_func = torchmetrics.F1Score(task="multiclass", num_classes=2)

    train_loop(
        train_data_loader,
        val_data_loader,
        args.epochs,
        model,
        loss_func,
        optimizer,
        args.out_dir,
        scheduler,
        metric_func,
        device,
    )


# training driver code
if __name__ == "__main__":
    # initalize various utilities

    args = parse_cli()

    # search_space = {
    #     "learning_rate": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    #     "weight_decay": tune.choice([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    #     "batch_size": tune.grid_search([2, 4, 8, 16, 32, 64]),
    #     "hidden_size": tune.grid_search([128, 256, 512]),
    #     "num_layers": tune.grid_search([1, 2, 3]),
    #     "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    # }

    # search_space = {
    #     "learning_rate": tune.grid_search([1e-3, 1e-2, 1e-1]),
    #     "weight_decay": tune.choice([1e-5]),
    #     "batch_size": tune.grid_search([32]),
    #     "hidden_size": tune.grid_search([128, 256, 512]),
    #     "num_layers": tune.grid_search([2, 3]),
    #     "dropout": tune.choice([0.3, 0.4, 0.5, 0.6]),
    # }

    search_space = {
        "learning_rate": tune.grid_search([1e-3, 1e-2]),
        "weight_decay": tune.choice([1e-5]),
        "batch_size": tune.grid_search([32]),
        "hidden_size": tune.grid_search([128, 256]),
        "num_layers": tune.grid_search([2, 3]),
        "dropout": tune.choice([0.5]),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1)
    
    trainable_with_resources = tune.with_resources(training_wrapper, {"cpu": 18,})
    # execute training
    tuner = tune.Tuner(
        trainable=trainable_with_resources,
        run_config=train.RunConfig(
            name=f"lstm_classifier_training_{str(time()).split('.')[0]}",
            storage_path=args.out_dir,
            # callbacks=[],
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="eval_metric",
                checkpoint_score_order="max",
                num_to_keep=5,
            ),     
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,
            max_concurrent_trials=1,
            scheduler=ASHAScheduler(metric="eval_metric", mode="max"),
        ),
        
        param_space=search_space,
    )

    result = tuner.fit()

    best_trial = result.get_best_trial("eval_metric", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["eval_loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["eval_metric"]))

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="eval_metric", mode="max")
    best_checkpoint_dir = best_checkpoint.to_directory(path=args.out_dir)
    # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    # best_trained_model = model._network
    # best_trained_model.load_state_dict(model_state)
    # model._network.load_state_dict(model_state)


    # save final model checkpoint
    # torch.save(
    #     {"model_state_dict": model.state_dict()},
    #     os.path.join(args.out_dir, "last.pt"),
    # )
