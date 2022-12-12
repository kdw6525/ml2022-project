# CrossViT

This repository is a clone of the official [CrossViT Implementation](https://github.com/IBM/CrossViT). 

Which implements the model from the [ArXiv](https://arxiv.org/abs/2103.14899) paper, and is cited in the writeup.

## Installation

To install requirements:

```setup
pip install -r requirements.txt
```

With conda:

```
conda create -n crossvit python=3.8
conda activate crossvit
conda install pytorch=1.7.1 torchvision  cudatoolkit=11.0 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data preparation

The CIFAR-10 data set will automatically be downloaded when the model is run.

## Pretrained models

We have 5 pretrained models included, they are in the checkpoint/X_X/ where X is the patch sizes of the model.
Details about each epoch is saved in the log.txt file in each directory.

## Training

The command we used to train a `crossvit_patch_test` on CIFAR-10 (CIFAR-10 will automatically be downloaded):

```shell script
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model crossvit_patch_test --batch-size 100 --epochs 200 --patch_size X,X --data-set CIFAR10 --data-path ./CIFAR10/ --output_dir checkpoints/X_X/ 
```

Resuming training requires adding one additional flag.

```shell script
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model crossvit_patch_test --batch-size 100 --epochs 200 --patch_size X,X --data-set CIFAR10 --data-path ./CIFAR10/ --output_dir checkpoints/X_X/ --auto-resume
```

## Evaluation

To evaluate a model:

```shell script
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model crossvit_patch_test --batch-size 100 --data-set CIFAR10 --data-path ./CIFAR10/ --patch_size X,X --initial_checkpoint checkpoints/X_X/model_best.pth --eval
```
