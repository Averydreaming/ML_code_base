# Project 0: Basics

## Goal

**Correctness** is the most important.

- Implement components in (Quant) ML pipeline

  	- data

  - model

  - evaluation

- Develop your basic benchmarks
  - linear model
  - neural network

## Data

- x: /gpfs/hddfs/shared/lshi_mli_hruan/project_0/x (1253d, feature_0001-feature_1252)
- y: /gpfs/hddfs/shared/lshi_mli_hruan/project_0/y (scalar, y5d)
- date range: 20170103 -- 20221230

## Code Structure

```
project/
│
├── information/
│   ├── X_feature_information.py
│   ├── Y_feature_information.py
│		└── date_information.py
│
├── data_loader.py
│
├── models/
│   ├── linear_regression.py
│   ├── mlp.py
│   ├── resnet.py
│   └── transformer.py
│
├── loss.py
│
├── metric/
│   ├── metric_Ypred.py
│   └── metric_between_Ypred_and_Ytrue.py
│
├── evaluation.py
│
├── single_window_main.py
├── main.py (add rolling window )
└── config.py

```



### User Guide

1. You can train a model (for single window experiment) by running:

   `python single_window_main.py --model="LR" --loss_fn="MSE" --optimizer="Adam" --batch_size=64 --epochs=10        --output_dir="LR_train320_val30_test10" --train_date_num=320 --val_date_num=30 --test_date_num=30 --mode="single_subset"`

2. You can train a model (for rolling window experiment) by running:

   `python main.py --model="LR" --loss_fn="MSE" --optimizer="Adam" --batch_size=64 --epochs=10 --output_dir="LR_train320_val30_test10" --train_date_num=320 --val_date_num=30 --test_date_num=30`

3. You can also evaluate outputs by running  `python evaluation.py`

   **There are other parameters that can be tuned. Here we describe them in detail:**

   | Parameter name | Description of parameter                                     |
   | :------------- | :----------------------------------------------------------- |
   | seed           | random seed                                                  |
   | date_dir       | the root path of X                                           |
   | label_dir      | the root path of Y                                           |
   | mode           | The selection of training modes is divided into the following three types: single window/global window/rolling |
   | train_date_num | Number of days in Training dataset (default to 500)          |
   | val_date_num   | Number of days in Validation dataset (default to 40)         |
   | test_date_num  | Number of days in Test dataset  (default to 20)              |
   | end_date       | the end day of prediction                                    |
   | rolling_stride | the step size of the moving window (default to 20)           |
   | X_norm_type    | Select the normalization method of X from None,min-max,z-score (default to min-max) |
   | Y_norm_type    | Select the normalization method of Y from None,min-max,z-score (default to min-max) |
   | model          | Choose the model from LR,MLP,Resnet (default to LR)          |
   | loss_fn        | Choose the loss function from MSE, Corr  (default to MSE)    |
   | epochs         | number of epochs of training  (default to 10)                |
   | learning_rate  | the initial learning rate for the optimizer (default to 1e-4) |
   | batch_size     | the batch size for training and testing  (default to 64)     |
   | num_workers    | the num_workers of Data loaders (default to 1)               |
   | input_dim      | the dim of X_features  (default to 1253)                     |
   | optimizer      | an method used to adjust the parameters of a model（default to Adam） |
   | output_dir     | the path of output                                           |

   

### Data visualization



