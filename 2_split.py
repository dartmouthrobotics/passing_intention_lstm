import os
import pandas as pd
import numpy as np
import yaml
from aux_code.learning_preprocess import get_trajectory_before_pass


np.random.seed(0)  # TODO check random seed

### Define paths
DATASET_ROOT = "./datasets"
FULL_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_full_dataset.parquet")
REAL_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_real_dataset.parquet")
SYNTHETIC_NOISE_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_synthetic_noise_dataset.parquet")
SYNTHETIC_NO_NOISE_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_synthetic_no_noise_dataset.parquet")

OUT_TRAIN_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_train_dataset.parquet")
OUT_VAL_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_val_dataset.parquet")
OUT_TEST_DATASET_PARQUET_PATH = os.path.join(DATASET_ROOT, "preprocessed_test_dataset.parquet")
CONFIG_PATH = "./param/lstm_config.yaml"


### load parquet
df_entire_pass = pd.read_parquet(FULL_DATASET_PARQUET_PATH)
df_real_pass = pd.read_parquet(REAL_DATASET_PARQUET_PATH) # 0-199
df_synthetic_noise_pass = pd.read_parquet(SYNTHETIC_NOISE_DATASET_PARQUET_PATH) # 200-1199
df_synthetic_no_noise_pass = pd.read_parquet(SYNTHETIC_NO_NOISE_DATASET_PARQUET_PATH) # 1200-2199

df_list = [df_real_pass, df_synthetic_noise_pass, df_synthetic_no_noise_pass]

# https://www.geeksforgeeks.org/how-to-randomly-select-elements-of-an-array-with-numpy-in-python/

# -----------------------------------------------------------------------------
# Main split part
# -----------------------------------------------------------------------------

### Load config
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
LABELS = config['LABELS']
TRAIN_RATIO = config['TRAIN_RATIO']
VAL_RATIO = config['VAL_RATIO']

print("LABELS:", LABELS)
print("TRAIN_RATIO:", TRAIN_RATIO)
print("VAL_RATIO:", VAL_RATIO)


### sampling split 
train_obj_id = np.array([])
val_obj_id = np.array([])
test_obj_id = np.array([])

for df_input in df_list:
    print("--------------------------------------------------")
    for label_ in LABELS: # L, R extraction
        # extract each side
        use_df_side = df_input.loc[(df_input.valid == True) & (df_input.label == label_)] # to be used, valid df

        unique_id_side = use_df_side.obj_index.unique()

        # data split
        train_data_size_side = int(len(unique_id_side) * TRAIN_RATIO)
        val_data_size_side = int(len(unique_id_side) * VAL_RATIO)
        test_data_size_side = len(unique_id_side) - train_data_size_side - val_data_size_side
        print("Label: {} \n train size: {} \n val szie: {} \n test size: {}".format(label_, 
                                                                                    train_data_size_side, 
                                                                                    val_data_size_side, 
                                                                                    test_data_size_side))

        # split obj indexes
        train_obj_id_side = np.random.choice(unique_id_side, size = train_data_size_side, replace=False)
        remaining_obj_id = np.setdiff1d(unique_id_side, train_obj_id_side)
        val_obj_id_side = np.random.choice(remaining_obj_id, size = val_data_size_side, replace=False)
        test_obj_id_side = np.setdiff1d(remaining_obj_id, val_obj_id_side)
        

        train_obj_id = np.append(train_obj_id,train_obj_id_side)
        val_obj_id = np.append(val_obj_id,val_obj_id_side)
        test_obj_id = np.append(test_obj_id,test_obj_id_side)

        # print("tran id", train_obj_id)
        # print("val_obj_id", val_obj_id)
        # print("test_obj_id", test_obj_id)

print("--------------------------------------------------")
print("final train size: {} \n final val size: {} \n final test size: {} \n total {}".format(len(train_obj_id), 
                                                                                            len(val_obj_id), 
                                                                                            len(test_obj_id), 
                                                                                            len(train_obj_id)+len(val_obj_id)+len(test_obj_id)
                                                                                            ))

# -------------------------

### Save final split data
# train data
df_cropped_train = get_trajectory_before_pass(df_entire_pass, train_obj_id)

assert np.all(df_cropped_train["label"])
assert not np.any(np.isnan(df_cropped_train["y"]))

df_cropped_train.to_parquet(OUT_TRAIN_DATASET_PARQUET_PATH)

# validation data
df_cropped_val = get_trajectory_before_pass(df_entire_pass, val_obj_id)

assert np.all(df_cropped_val["label"])
assert not np.any(np.isnan(df_cropped_val["y"]))

df_cropped_val.to_parquet(OUT_VAL_DATASET_PARQUET_PATH)

# test data
df_cropped_test = get_trajectory_before_pass(df_entire_pass, test_obj_id)

assert np.all(df_cropped_test["label"])
assert not np.any(np.isnan(df_cropped_test["y"]))

df_cropped_test.to_parquet(OUT_TEST_DATASET_PARQUET_PATH)
