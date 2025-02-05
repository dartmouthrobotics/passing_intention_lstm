import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn

import yaml
import torchmetrics
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

dataset_ = __import__("3_dataset")
model_ = __import__("4_model")

MODEL_PATH = "./weights/model.pth"
TRAIN_PARQUET_PATH = "./datasets/preprocessed_train_dataset.parquet"
TEST_PARQUET_PATH = "./datasets/preprocessed_test_dataset.parquet"
PLOT_SAVE_PATH = "./imgs/conf_mat.png"

CONFIG_PATH = "./param/lstm_config.yaml"

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
OBSERVATION_LENGTH = config['OBSERVATION_LENGTH']
num_features_ = config['num_features']
num_classes_ = config['num_classes']
hidden_size_ = config['hidden_size']
num_layers_ = config['num_layers']
dropout_fraction_ = config['dropout_fraction']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_func = torchmetrics.F1Score(task="multiclass", num_classes=num_classes_)
metric_func.to(device=device)


model = model_.TimeSeriesClassifier(num_features=num_features_,
                                    num_classes=num_classes_, 
                                    hidden_size=hidden_size_, 
                                    num_layers=num_layers_,
                                    )

checkpoint = torch.load(MODEL_PATH)
for key in checkpoint:
    print(key)
# model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict']) # .pt 
model.load_state_dict(torch.load(MODEL_PATH)) # pth



print(model)
model.to(device=device)
model.eval()

# row_dim = max(
#     pd.read_parquet(TEST_PARQUET_PATH).groupby("obj_index").size().max(),
#     pd.read_parquet(TRAIN_PARQUET_PATH).groupby("obj_index").size().max(),
# )

row_dim = OBSERVATION_LENGTH
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
print("Test dataset size:", len(test_dataloader))

preds = []
gt = []
with torch.no_grad():
    for X, y, l in test_dataloader:
        # X = pack_padded_sequence(X, lengths=l, batch_first=True, enforce_sorted=False)
        X, y = X.to(device=device), y.to(device=device)
        # get model guess
        logits = model(X)
        # post-process guess
        softmax_ = nn.Softmax(dim=1)
        softmax_logits_ = softmax_(logits)
        _, preds_ = torch.max(softmax_logits_, 1) 
        _, gt_ = torch.max(y, 1) 
        preds.append(preds_.cpu())
        gt.append(gt_.cpu())
    metric = metric_func(torch.cat(preds), torch.cat(gt))
    print("Test Dataset F1 Score of ", metric.cpu().item())

mat = confusion_matrix(gt, preds)
disp = ConfusionMatrixDisplay(mat, display_labels=test_dataset.classes_dict)
disp.plot()
plt.savefig(PLOT_SAVE_PATH)
