# Neural Network Training for Active Learning-augmented Intent-aware Obstacle Avoidance of Autonomous Surface Vehicles in High-traffic Waters Paper

* This repository contains the code for our LSTM-backbone neural network training to predict the side a vessel will pass: left or right to your vessel as well as data generator including real-world AIS / synthetic traffic.
* Submitted to `ICRA 2024` by Dartmouth Robotics

## Contributors
* Ari Chadda
* Mingi Jeong
* Alberto Quattrini Li

<img src="./imgs/demo.gif" width="800">


## Tested Environment & Dependencies
* Ubuntu 20.04
* Python 3.10

For a description of Python dependencies, please see the `pyproject.toml` file.

Dependencies are managed using [poetry](https://python-poetry.org/). You will need to install poetry, see these [instructions](https://python-poetry.org/docs/), we used `poetry==1.6.1`.



## Folder structure
```
├── 1_preprocessing.py
├── 2_split.py
├── 3_dataset.py
├── 4_model.py
├── 5_train.py
├── 6_eval.py
├── aux_code
│   └── learning_preprocess.py
├── datasets
│   ├── extracted_data
│   ├── preprocessed_full_dataset.parquet
│   ├── preprocessed_test_dataset.parquet
│   ├── preprocessed_train_dataset.parquet
│   └── synthetic_data
├── notebooks
│   ├── AIS_based_LSTM-preprocess.ipynb
│   └── AIS-synthetic-generator.ipynb
├── poetry.lock
├── pyproject.toml
├── README.md
└── weights
    └── best.pt

```

## Usage Instructions
To run, clone this repository and then run from the passing_intention directory:

### 1. Data Preprocess
In `notebooks`, you can run script to generate `.pkl` files for passing traffic ground truth data.
For real-world AIS, please see the following Danish Maritime [open source dataset](https://dma.dk/safety-at-sea/navigational-information/ais-data) and download `.csv` accordingly. 


### 2. LSTM Training 
The overall relationship of the files in this directory can be seen below.

```mermaid
flowchart TD
    preprocess(1_preprocess.py) --> split(2_split.py)
    split(2_split.py) --> train(5_train.py)
    dataset(3_dataset.py) --> train(5_train.py)
    model(4_model.py) --> train(5_train.py)
    train(5_train.py) --> eval(6_eval.py)
```

To run, please use the following commands from the root of this repository

```bash
poetry shell # create virtualenv
poetry install # install project dependencies
python 1_preprocessing.py # prepare data for training
python 2_split.py # create train/test split
python 5_train.py --parquet-path-train ./datasets/preprocessed_train_dataset.parquet --parquet-path-test ./datasets/preprocessed_test_dataset.parquet --learning-rate 1e-2 --num-workers 10 --is-training True --epochs 10000 --batch-size 20 # stopped at approx. 1000 epochs
python 6_eval.py # evaluate results
```

<img src="./imgs/overall.png" width="500">
<img src="./imgs/architecture.png" width="500">
