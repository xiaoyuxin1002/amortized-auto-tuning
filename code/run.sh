#!/bin/bash


profile_id=0
    
num_inducing_point=1000
batch_size=1000

num_training_epoch=200
lr=0.02
momentum=0.8
warmup_ratio=0.01

beta=0.25
num_tuning_budget=100
num_tuning_epoch=3


for database in 'HyperRec' 'LCBench'
do
    for pair_id in 0 1 2 3 4
    do
        python3 main.py --database=${database} --profile_id=${profile_id} --pair_id=${pair_id} --num_inducing_point=${num_inducing_point} --batch_size=${batch_size} --num_training_epoch=${num_training_epoch} --lr=${lr} --momentum=${momentum} --warmup_ratio=${warmup_ratio} --beta=${beta} --num_tuning_budget=${num_tuning_budget} --num_tuning_epoch=${num_tuning_epoch} 
    done
done