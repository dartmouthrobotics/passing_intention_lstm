# Passing Intention LSTM Training

## LSTM Training for ___ Paper
This repository contains the code for LSTM training to predict the side a vessel will pass: left or right to your vessel. 

## Tested environment
* Ubuntu 20.04
* Python 3.10

## Dependencies
Dependencies are managed using [poetry](https://python-poetry.org/). To run, clone this repository and then run from the passing_intetion directory: 

```bash
poetry shell # create virtualenv
poetry install # install project deps
python 1_preprocessing.py # prepare data for training
python 2_split.py # create train/test split
python 5_train.py # train model
python 6_eval.py # evaluate results
python 7_tune.py # tune hyperparameters 
```