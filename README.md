# DistCare: Distilling Knowledge from Publicly Available Online EMR Data to Emerging Epidemic for Prognosis
Code for *DistCare: Distilling Knowledge from Publicly Available Online EMR Data to Emerging Epidemic for Prognosis* (WWW '21)

Thanks for your interest in our work!

## Requirements

- Python: version 3.0+. We use Python 3.7.3.
- Pytorch: version 1.1+. We use Pytorch 1.1.
- CUDA is needed if you want to train with GPUs.

## Data Preparation

We do not provide the PhysioNet, Tongji Hospital, and Spain HM Hospital datasets themselves. The introduction of these datasets can be found as follows:

- PhysioNet Challenge: http://physionet.org/content/challenge-2019/1.0.0/
- Tongji Hospital COVID-19: https://www.nature.com/articles/s42256-020-0180-7
- Spain HM Hospital COVID-19: We are sorry that this dataset is not publicly available, and we cannot provide the dataset due to privacy agreements and copyright policies.

Also, data preprocessing is needed. We use python lib ```pickle``` to dump and load data from files.

## Run DistCare

All the hyperparameters are provided in our code ```DistCare.ipynb```. You can run it directly.

If you don't want to pretrain on PhysioNet, we've already provided pretrained parameters for source (PhysioNet) dataset. You can simply load them to model, and run the transfer learning experiment on target dataset (Tongji or Spain).
