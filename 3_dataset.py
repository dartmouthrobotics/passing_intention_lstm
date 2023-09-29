import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List, Dict


class PassingIntentionDataset(Dataset):
    X_COLS = [
        "x",
        "y",
        "rel_dist",
        "heading_converted",
        "sin_rel_bearing",
        "cos_rel_bearing",
    ]
    Y_COLS = ["label"]
    CLASSES_DICT = {"L": 0, "R": 1}

    def __init__(
        self,
        parquet_path: str,
        row_dim: int,
        X_cols: List[str] = X_COLS,
        y_cols: List[str] = Y_COLS,
        classes_dict: Dict[int, str] = CLASSES_DICT,  # binary classification by default
    ):
        self.df = pd.read_parquet(parquet_path)
        self.row_dim = row_dim

        self.X_cols = X_cols
        self.y_cols = y_cols
        self.n_classes = len(classes_dict)
        self.classes_dict = classes_dict

        self.data_array = []
        self.label_array = []

        for _, df_by_ID in self.df.groupby("obj_index"):
            X = torch.tensor(df_by_ID[self.X_cols].values.tolist())
            X = F.pad(X, pad=(0, 0, 0, row_dim - X.shape[0]))
            self.data_array.append(X)

            y = [0.0] * len(self.classes_dict)
            y[self.classes_dict[df_by_ID[self.y_cols].values[0].item()]] = 1.0
            self.label_array.append(torch.tensor(y))

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, idx):
        X = self.data_array[idx]
        y = self.label_array[idx]

        return X, y


if __name__ == "__main__":
    ### Usage example
    TRAIN_PARQUET_PATH = (
        "./datasets/preprocessed_train_dataset.parquet"
    )
    TEST_PARQUET_PATH = (
        "./datasets/preprocessed_test_dataset.parquet"
    )

    row_dim = max(
        pd.read_parquet(TEST_PARQUET_PATH).groupby("obj_index").size().max(),
        pd.read_parquet(TRAIN_PARQUET_PATH).groupby("obj_index").size().max(),
    )

    train_dataset = PassingIntentionDataset(
        parquet_path=TRAIN_PARQUET_PATH, row_dim=row_dim
    )
    test_dataset = PassingIntentionDataset(
        parquet_path=TEST_PARQUET_PATH, row_dim=row_dim
    )
