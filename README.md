# Amortized Auto-Tuning (AT2) Method

AT2 is a multi-task multi-fidelity Bayesian optimization approach. \
It leverages cheap-to-obtain low-fidelity tuning observations to achieve cost-efficient hyperparameter transfer optimization.

## Requirements

```
PyTorch = 1.8.1
GPyTorch = 1.4.2
```

## Data

The ```data``` folder contains the two databases and the five train-test task pairs for each database we used in our experiments.

## Run

The set of training hyperparameters are specified in ```code/run.sh```. \
To run our sweep of experiments, use the command ```bash code/run.sh```.

## Result

The standard output of the training process and the training results will be stored in the ```record``` folder.

# Hyperparameter Recommendation (HyperRec) Database

HyperRec is a hyperparameter recommendation database for image classification tasks. \
It consists of 27 unique image classification tasks and 150 distinct configurations sampled from a 16-dimensional nested hyperparameter space.

Users can retrieve it <a href="https://drive.google.com/drive/folders/1M6SF-T0LJqLMuO-N_0dFSCGfnesqK5J3?usp=sharing">here</a>.

## Tasks

The statistics of the 27 image classification tasks are as follows:

**Task/Dataset** | ACTION40 | AWA2 | BOOKCOVER30 | CALTECH256 | CARS196 | CIFAR10 | CIFAR100 | CUB200 | FLOWER102 | FOOD101 | IMAGENET64SUB1 | IMAGENET64SUB2 | IMAGENETSUB3 | IP102 | ISR | OIPETS | PLACE365SUB1 | PLACE365SUB2 | PLACE365SUB3 | PLANT39 | RESISC45 | SCENE15 | SDD | SOP | SUN397SUB1 | SUN397SUB2 | SUN397SUB3 
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
**Number of Images** | 9,532 | 37,322 | 57,000 | 30,607 | 16,185 | 60,000| 60,000 | 11,788 | 8,189 | 101,000 | 128,112 | 128,112 | 128,112 | 75,222 | 15,620 | 7,349 | 91,987 | 91,987 | 91,987 | 61,486 | 31,500 | 4,485 | 20,580 | 120,053 | 9,925 | 9,925 | 9,925
**Number of Classes** | 40 | 50 | 30 | 257 | 196 | 10 | 100 | 200 | 102 | 101 | 1,000 | 1,000 | 1,000 | 102 | 67 | 37 | 365 | 365 | 365 | 39 | 45 | 15 | 120 | 12 | 397 | 397 | 397

The original image classification dataset of each task is split based on a common ratio: 60\% for the training set, 20\% for the validation set, and 20\% for the testing set. 

For each task, we evaluate each configuration during 75 training epochs and repeat this with 2 randomly sampled seeds.

During training, we record the following information for the training set:
- Batch-wise cross-entropy loss
- Batch-wise top one, five, and ten accuracies
- Epoch-wise training time

During evaluation, we record the following information for the validation and testing sets separately:
- Epoch-wise cross-entropy loss 
- Epoch-wise top one, five, and ten accuracies
- Epoch-wise evaluation time 

## Hyperparameter Space

The following notations represent the sampling distributions used in the 16-dimensional hyperparameter space:
- *C*{ } denotes the categorical distribution
- *U*( , ) denotes the uniform distribution
- *U*{ , } denotes the discrete uniform distribution
- *LU*( , ) denotes the log-uniform distribution

Some of the hyperparameters are independent of any categorical variables:

**Hyperparameter** | Tuning Distribution 
--- | ---
Batch size | *U*{32, 128}
Model | *C*{ResNet34, ResNet50}
Optimizer | *C*{Adam, Momentum}
Learning Rate Scheduler | *C*{StepLR, ExponentialLR, CyclicLR, CosineAnnealingWarmRestarts}

Some of the hyperparameters are dependent of the choice of optimizer or learning rate scheduler:

**Optimizer Choice** | Hyperaparameter | Tuning Distribution 
--- | --- | ---
Adam | Learning rate<br>Weight decay<br>Beta_0<br>Beta_1 | *LU*(1e-4, 1e-1)<br>*LU*(1e-5, 1e-3)<br>*LU*(0.5, 0.999)<br>*LU*(0.8, 0.999)
Momentum | Learning rate<br>Weight decay<br>Momentum factor | *LU*(1e-4, 1e-1)<br>*LU*(1e-5, 1e-3)<br>*LU*(1e-3, 1)

**Learning Rate Scheduler Choice** | Hyperaparameter | Tuning Distribution 
--- | --- | ---
StepLR | Step size<br>Gamma | *U*{2, 20}<br>*LU*(0.1, 0.5)
ExponentialLR | Gamma | *LU*(0.85, 0.999)
CyclicLR | Gamma<br>Max learning rate<br>Step size up | *LU*(0.1, 0.5)<br>min(1, learning rate * *U*(1.1, 1.5))<br>*U*{1, 10}
CosineAnnealingWarmRestarts | T_0<br>T_mult<br>Eta_min | *U*{2, 20}<br>*U*{1, 4}<br>learning rate * *U*{0.5, 0.9}


# Citation

Please cite the following work if you find the AT2 method or the HyperRec database useful.
```
@misc{xiao2021amortized,
    title={Amortized Auto-Tuning: Cost-Efficient Transfer Optimization for Hyperparameter Recommendation},
    author={Yuxin Xiao and Eric P. Xing and Willie Neiswanger},
    year={2021},
    eprint={2106.09179},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
