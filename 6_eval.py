import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchmetrics

dataset_ = __import__("3_dataset")
model_ = __import__("4_model")

MODEL_PATH = "/home/arichadda/modeling/passing_intention/passing_intention_model_training/1694167331/best.pt"
TRAIN_PARQUET_PATH = "./datasets/preprocessed_train_dataset.parquet"
TEST_PARQUET_PATH = "./datasets/preprocessed_test_dataset.parquet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_func = torchmetrics.F1Score(task="multiclass", num_classes=2)
metric_func.to(device=device)

model = model_.TimeSeriesClassifier(num_features=6, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
model.to(device=device)
model.eval()

row_dim = max(
    pd.read_parquet(TEST_PARQUET_PATH).groupby("obj_index").size().max(),
    pd.read_parquet(TRAIN_PARQUET_PATH).groupby("obj_index").size().max(),
)

test_dataset = dataset_.PassingIntentionDataset(
    parquet_path=TEST_PARQUET_PATH, row_dim=row_dim
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)

preds = []
gt = []
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device=device), y.to(device=device)
        # get model guess
        logits = model(X)
        # post-process guess
        _, preds_ = torch.max(logits, 1)
        _, gt_ = torch.max(y, 1)
        preds.append(preds_)
        gt.append(gt_)
    metric = metric_func(torch.cat(preds), torch.cat(gt))
    print("F1 Score of ", metric)
