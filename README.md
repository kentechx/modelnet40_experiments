# modelnet40_experiments

This repository contains experiments conducted on the ModelNet40 dataset. The ModelNet40 dataset is a widely used
benchmark dataset for 3D classification tasks.

## Dataset

The ModelNet40 dataset consists of 12,311 CAD models from 40 object categories, with approximately 300 instances per
category. The dataset is divided into two sets: a training set of 9,843 models and a test set of 2,468 models. The
original CAD models are provided in the `.off` format.

The experiments are conducted on the processed data in `.hdf5` format. Each processed object contains 2048 points
sampled from the original CAD models.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Experiments

The experiments are conducted on the following models:

- [DGCNN](https://github.com/kentechx/x-dgcnn)
- [PointNet](https://github.com/kentechx/pointnet)
- [PointNet2](https://github.com/kentechx/pointnet)
- [PointNext](https://github.com/kentechx/pointnext)

Training models by running the corresponding scripts in the `code` folder. For example, to train the DGCNN model, run
the following command:

```bash
python code/train_dgcnn.py
```

## Results

The table below presents the classification accuracy of the models on the ModelNet40 dataset (1024 points, trained on a
single Nvidia RTX 3090 GPU). Some of the following results are the best outcomes obtained after hyperparameter
sweepings.

| Model        | input | Overall Accuracy | sweep   |
|--------------|-------|------------------|---------|
| DGCNN        | xyz   | 91.7%            | &check; |
| PointNet     | xyz   | 90.7%            | &check; |
| PointNet2SSG | xyz   | 90.7%            |         |
| PointNet2MSG | xyz   | 92.1%            |         |
| PointNext    | xyz   | 91.3%            |         |

You can reproduce the results by running the corresponding scripts in the `code` folder with default configurations.
For example, to train the PointNet model, run the following command

```bash
python code/train_pointnet.py
```
