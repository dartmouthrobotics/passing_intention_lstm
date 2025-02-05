import torch
import pandas as pd
import yaml
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import List, Dict

CONFIG_PATH = "./param/lstm_config.yaml"

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
OBSERVATION_LENGTH = config['OBSERVATION_LENGTH']
X_COLS = config['X_COLS']
Y_COLS = config['Y_COLS']
CLASSES_DICT = config['CLASSES_DICT']

class PassingIntentionDataset(Dataset):
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
        self.length_array = []
        self.label_array = []

        # based on observation length, we generate targets
        # therefore, parquet is always fixed.
        for _, df_by_ID in self.df.groupby("obj_index"):
            for i in range(0, len(df_by_ID) - (OBSERVATION_LENGTH-1)):
                # 1) X_ data by iterating observation rows
                sub_df_by_ID = df_by_ID.iloc[i:i+OBSERVATION_LENGTH]
                X = torch.tensor(sub_df_by_ID[self.X_cols].values.tolist())
                
                self.length_array.append(len(X))
                # https://discuss.pytorch.org/t/visual-explanation-of-torch-pad/96999/2
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
        l = self.length_array[idx]

        return X, y, l


if __name__ == "__main__":
    ### Usage example
    TRAIN_PARQUET_PATH = "./datasets/preprocessed_train_dataset.parquet"
    VAL_PARQUET_PATH = "./datasets/preprocessed_val_dataset.parquet"
    TEST_PARQUET_PATH = "./datasets/preprocessed_test_dataset.parquet"

    # no need if we use fixed observation length (matching ros length)
    # row_dim = max(
    #     pd.read_parquet(TEST_PARQUET_PATH).groupby("obj_index").size().max(),
    #     pd.read_parquet(TRAIN_PARQUET_PATH).groupby("obj_index").size().max(),
    # )

    row_dim = OBSERVATION_LENGTH

    train_dataset = PassingIntentionDataset(parquet_path=TRAIN_PARQUET_PATH, row_dim=row_dim)
    test_dataset = PassingIntentionDataset(parquet_path=TEST_PARQUET_PATH, row_dim=row_dim)
    val_dataset = PassingIntentionDataset(parquet_path=VAL_PARQUET_PATH, row_dim=row_dim)


    #### Data size check (batch size 1)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # log size to console
    print("TRAIN DATALOADER LENGTH:", len(train_dataloader))
    print("TEST DATALOADER LENGTH:", len(test_dataloader))
    print("VAL DATALOADER LENGTH:", len(val_dataloader))
