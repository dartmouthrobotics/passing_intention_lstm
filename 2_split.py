import os
import pandas as pd
import numpy as np
from aux_code.learning_preprocess import get_trajectory_before_pass


np.random.seed(0)

DATASET_ROOT = "/home/arichadda/modeling/passing_intention/datasets"
FULL_DATASET_PARQUET_PATH = os.path.join(
    DATASET_ROOT, "preprocessed_full_dataset.parquet"
)

OUT_TRAIN_DATASET_PARQUET_PATH = os.path.join(
    DATASET_ROOT, "preprocessed_train_dataset.parquet"
)
OUT_TEST_DATASET_PARQUET_PATH = os.path.join(
    DATASET_ROOT, "preprocessed_test_dataset.parquet"
)

df_entire_pass = pd.read_parquet(FULL_DATASET_PARQUET_PATH)

# https://www.geeksforgeeks.org/how-to-randomly-select-elements-of-an-array-with-numpy-in-python/

# for train, test data
use_df = df_entire_pass.loc[(df_entire_pass.valid == True)]  # to be used, valid df
unique_id = use_df.obj_index.unique()

# data split
train_data_size = int(len(unique_id) * 0.8)
test_data_size = len(unique_id) - train_data_size

# split obj indexes
train_obj_id = np.random.choice(unique_id, size=train_data_size, replace=False)
test_obj_id = np.setdiff1d(unique_id, train_obj_id)
train_obj_id

df_cropped_train = get_trajectory_before_pass(df_entire_pass, train_obj_id)
# df_cropped_train = df_cropped_train[
#     df_cropped_train["Heading_by_xy"].notna()
# ]  # drop na for Heaing_by_xy

assert np.all(df_cropped_train["label"])
assert not np.any(np.isnan(df_cropped_train["y"]))
# assert not np.any(np.isnan(df_cropped_train["Heading_by_xy"]))

df_cropped_train.to_parquet(OUT_TRAIN_DATASET_PARQUET_PATH)

df_cropped_test = get_trajectory_before_pass(df_entire_pass, test_obj_id)
# df_cropped_test = df_cropped_test[
#     df_cropped_test["Heading_by_xy"].notna()
# ]  # drop na for Heaing_by_xy

assert np.all(df_cropped_test["label"])
assert not np.any(np.isnan(df_cropped_test["y"]))
# assert not np.any(np.isnan(df_cropped_test["Heading_by_xy"]))

df_cropped_test.to_parquet(OUT_TEST_DATASET_PARQUET_PATH)
