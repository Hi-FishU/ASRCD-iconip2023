# ASRCD: Adaptive Serial Relation-Based Model for Cognitive Diagnosis

This is the source code for publication of ICONIP 2023 **ASRCD: Adaptive Serial Relation-Based Model for Cognitive Diagnosis** (**[cite](https://doi.org/10.1007/978-981-99-8181-6_41)**)

## Getting Started

These instructions will get you a copy of ASRCD and running on your local machine for reaserch and testing purposes. See deployment for notes on how to deploy the model on a live system.

### Prerequisites

The things you need before deploy the model.

* PyTorch
* Scipy
* Pandas
* tqdm
* RayTune (If fine-tune needed)


## Usage

You could freely adjust the hyperparameters from the files shown in follows to fit your requirement.

* Manual train and test
```
$ src/train.py 
```
* Tuning by RayTune
```
$ src/tune_for_revised.py
```

## Dataset

The SPOC deidentifical dataset is published on Kaggle (**[link](https://www.kaggle.com/datasets/zhuonanliang/ms-namc)**).
