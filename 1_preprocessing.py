import os
import numpy as np
import pandas as pd

from aux_code.learning_preprocess import convert_from_NED_to_Robotic

REAL_DATA_ROOT = "./datasets/extracted_data"
REAL_FILE_PREFIX = "aisdk"

SYNTHETIC_DATA_ROOT = (
    "./datasets/synthetic_data/"
)
SYNTHETIC_DATA_NOISE_PATH = os.path.join(SYNTHETIC_DATA_ROOT, "noise")
SYNTHETIC_DATA_NO_NOISE_PATH = os.path.join(SYNTHETIC_DATA_ROOT, "no_noise")

SYNTHETIC_FILE_PREFIX = "synthetic"

OUT_PATH = "./datasets/preprocessed_full_dataset.parquet"

REAL_DATA_SENSOR_RANGE = 3000  # m
SYNTHETIC_DATA_SENSOR_RANGE = 200  # m
NORM_COLUMN_NAMES_LS = ["x", "y", "rel_dist"] 

def load_pkl_to_df(pkl_dir_path: str, pkl_file_prefix: str) -> pd.DataFrame:
    df = pd.DataFrame()
    count = len([x for x in os.listdir(pkl_dir_path) if x.startswith(pkl_file_prefix)])

    for idx in range(0, count):
        unpickled_df = pd.read_pickle(
            os.path.join(pkl_dir_path, f"{pkl_file_prefix}_{str(idx)}.pkl")
        )
        df = pd.concat([df, unpickled_df])
    return df


real_df_entire_pass = load_pkl_to_df(
    pkl_dir_path=REAL_DATA_ROOT, pkl_file_prefix=REAL_FILE_PREFIX
)
synthetic_noise_df = load_pkl_to_df(
    pkl_dir_path=SYNTHETIC_DATA_NOISE_PATH, pkl_file_prefix=SYNTHETIC_FILE_PREFIX
)
synthetic_no_noise_df = load_pkl_to_df(
    pkl_dir_path=SYNTHETIC_DATA_NO_NOISE_PATH, pkl_file_prefix=SYNTHETIC_FILE_PREFIX
)

real_df_entire_pass["heading_converted"] = np.deg2rad(real_df_entire_pass["Heading"])
# https://stackoverflow.com/questions/71249186/applying-function-to-column-in-a-dataframe
real_df_entire_pass["heading_converted"] = real_df_entire_pass[
    "heading_converted"
].apply(convert_from_NED_to_Robotic)

real_df_entire_pass[NORM_COLUMN_NAMES_LS] = (
    real_df_entire_pass[NORM_COLUMN_NAMES_LS] / REAL_DATA_SENSOR_RANGE
)

synthetic_noise_df["heading_converted"] = np.deg2rad(synthetic_noise_df["Heading"])
# https://stackoverflow.com/questions/71249186/applying-function-to-column-in-a-dataframe
synthetic_noise_df["heading_converted"] = synthetic_noise_df[
    "heading_converted"
].apply(convert_from_NED_to_Robotic)
synthetic_no_noise_df["heading_converted"] = np.deg2rad(synthetic_no_noise_df["Heading"])
# https://stackoverflow.com/questions/71249186/applying-function-to-column-in-a-dataframe
synthetic_no_noise_df["heading_converted"] = synthetic_no_noise_df[
    "heading_converted"
].apply(convert_from_NED_to_Robotic)

synthetic_noise_df[NORM_COLUMN_NAMES_LS] = (
    synthetic_noise_df[NORM_COLUMN_NAMES_LS] / SYNTHETIC_DATA_SENSOR_RANGE
)

synthetic_no_noise_df[NORM_COLUMN_NAMES_LS] = (
    synthetic_no_noise_df[NORM_COLUMN_NAMES_LS] / SYNTHETIC_DATA_SENSOR_RANGE
)

### Add synthetic data
assert len(real_df_entire_pass.groupby("obj_index")) == 200
assert len(synthetic_noise_df.groupby("obj_index")) == 1000
assert len(synthetic_no_noise_df.groupby("obj_index")) == 1000

df_entire_pass = real_df_entire_pass.reset_index()
synthetic_noise_df["obj_index"] = synthetic_noise_df["obj_index"] + 200
synthetic_no_noise_df["obj_index"] = synthetic_no_noise_df["obj_index"] + 1200
df_entire_pass = pd.concat([df_entire_pass, synthetic_noise_df, synthetic_no_noise_df])
df_entire_pass = df_entire_pass.reset_index()

assert len(df_entire_pass.groupby("obj_index")) == 2200
assert len(df_entire_pass) == len(real_df_entire_pass) + len(synthetic_noise_df) + len(
    synthetic_no_noise_df
)

df_entire_pass.to_parquet(OUT_PATH)
