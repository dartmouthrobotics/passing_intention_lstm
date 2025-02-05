import os
import numpy as np
import pandas as pd
import yaml
from aux_code.learning_preprocess import convert_from_NED_to_Robotic, get_heading_converted

REAL_DATA_ROOT = "./datasets/extracted_data"
REAL_FILE_PREFIX = "aisdk"
REAL_FILE_POSTFIX = "_clear"

SYNTHETIC_DATA_ROOT = "./datasets/synthetic_data/"
SYNTHETIC_DATA_NOISE_PATH = os.path.join(SYNTHETIC_DATA_ROOT, "noise")
SYNTHETIC_DATA_NO_NOISE_PATH = os.path.join(SYNTHETIC_DATA_ROOT, "no_noise")

SYNTHETIC_FILE_PREFIX = "synthetic"
SYNTHETIC_FILE_POSTFIX = "_clear"

OUT_PATH = "./datasets/preprocessed_full_dataset.parquet"
OUT_PATH_REAL = "./datasets/preprocessed_real_dataset.parquet"
OUT_PATH_SYNTHETIC_NOISE = "./datasets/preprocessed_synthetic_noise_dataset.parquet"
OUT_PATH_SYNTHETIC_NO_NOISE = "./datasets/preprocessed_synthetic_no_noise_dataset.parquet"
CONFIG_PATH = "./param/lstm_config.yaml"


def load_pkl_to_df(pkl_dir_path: str, pkl_file_prefix: str, pkl_file_postfix: str) -> pd.DataFrame:
    df = pd.DataFrame()
    count = len(
        [
            x
            for x in os.listdir(pkl_dir_path)
            if x.startswith(pkl_file_prefix) and pkl_file_postfix in x
        ]
    )

    for idx in range(0, count):
        unpickled_df = pd.read_pickle(
            os.path.join(pkl_dir_path, f"{pkl_file_prefix}_{str(idx)}{pkl_file_postfix}.pkl")
        )
        df = pd.concat([df, unpickled_df])
    return df

# -----------------------------------------------------------------------------
# Data load and preprocess 
# -----------------------------------------------------------------------------

### Load config
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Access the parameters
REAL_DATA_SENSOR_RANGE = config['REAL_DATA_SENSOR_RANGE']
SYNTHETIC_DATA_SENSOR_RANGE = config['SYNTHETIC_DATA_SENSOR_RANGE']
REAL_DATA_SIZE = config['REAL_DATA_SIZE']
SYNTHETIC_DATA_SIZE = config['SYNTHETIC_DATA_SIZE']
NORM_COLUMN_NAMES_LS = config['NORM_COLUMN_NAMES_LS']

# Print the loaded values to verify
print("REAL_DATA_SENSOR_RANGE:", REAL_DATA_SENSOR_RANGE)
print("SYNTHETIC_DATA_SENSOR_RANGE:", SYNTHETIC_DATA_SENSOR_RANGE)
print("REAL_DATA_SIZE:", REAL_DATA_SIZE)
print("SYNTHETIC_DATA_SIZE:", SYNTHETIC_DATA_SIZE)
print("NORM_COLUMN_NAMES_LS:", NORM_COLUMN_NAMES_LS)

### Data load
real_df_entire_pass = load_pkl_to_df(
    pkl_dir_path=REAL_DATA_ROOT,
    pkl_file_prefix=REAL_FILE_PREFIX,
    pkl_file_postfix=REAL_FILE_POSTFIX,
) # real-world data load (up to clear)
synthetic_noise_df = load_pkl_to_df(
    pkl_dir_path=SYNTHETIC_DATA_NOISE_PATH,
    pkl_file_prefix=SYNTHETIC_FILE_PREFIX,
    pkl_file_postfix=SYNTHETIC_FILE_POSTFIX,
) # synthetic data (noise) load
synthetic_no_noise_df = load_pkl_to_df(
    pkl_dir_path=SYNTHETIC_DATA_NO_NOISE_PATH,
    pkl_file_prefix=SYNTHETIC_FILE_PREFIX,
    pkl_file_postfix=SYNTHETIC_FILE_POSTFIX,
) # synthetic data (no noise) load

# print(real_df_entire_pass.columns)

### pre-process
# real-world data
# heading convert (NED -> robotic)
real_df_entire_pass = get_heading_converted(input_df=real_df_entire_pass)
# normalize sensor range -> vehicle agnostic
real_df_entire_pass[NORM_COLUMN_NAMES_LS] = (
    real_df_entire_pass[NORM_COLUMN_NAMES_LS] / REAL_DATA_SENSOR_RANGE
)

# synthetic data
# heading convert (NED -> robotic)
synthetic_noise_df = get_heading_converted(input_df=synthetic_noise_df)
synthetic_no_noise_df = get_heading_converted(input_df=synthetic_no_noise_df)
# normalize sensor range -> vehicle agnostic
synthetic_noise_df[NORM_COLUMN_NAMES_LS] = (
    synthetic_noise_df[NORM_COLUMN_NAMES_LS] / SYNTHETIC_DATA_SENSOR_RANGE
)
synthetic_no_noise_df[NORM_COLUMN_NAMES_LS] = (
    synthetic_no_noise_df[NORM_COLUMN_NAMES_LS] / SYNTHETIC_DATA_SENSOR_RANGE
)


### Add synthetic data and entire_df check
assert len(real_df_entire_pass.groupby("obj_index")) == REAL_DATA_SIZE
assert len(synthetic_noise_df.groupby("obj_index")) == SYNTHETIC_DATA_SIZE
assert len(synthetic_no_noise_df.groupby("obj_index")) == SYNTHETIC_DATA_SIZE

df_entire_pass = real_df_entire_pass.reset_index()
synthetic_noise_df["obj_index"] = synthetic_noise_df["obj_index"] + REAL_DATA_SIZE
synthetic_no_noise_df["obj_index"] = (
    synthetic_no_noise_df["obj_index"] + REAL_DATA_SIZE + SYNTHETIC_DATA_SIZE
)
df_entire_pass = pd.concat([df_entire_pass, synthetic_noise_df, synthetic_no_noise_df])
df_entire_pass = df_entire_pass.reset_index()

assert len(df_entire_pass.groupby("obj_index")) == REAL_DATA_SIZE + 2 * SYNTHETIC_DATA_SIZE
assert len(df_entire_pass) == len(real_df_entire_pass) + len(synthetic_noise_df) + len(
    synthetic_no_noise_df
)

### Save the processed data into parquet
df_entire_pass.to_parquet(OUT_PATH)
real_df_entire_pass.to_parquet(OUT_PATH_REAL)
synthetic_noise_df.to_parquet(OUT_PATH_SYNTHETIC_NOISE)
synthetic_no_noise_df.to_parquet(OUT_PATH_SYNTHETIC_NO_NOISE)

print("pre processed data have been saved...")
