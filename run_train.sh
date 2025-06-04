#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
NODE='node-name'
NAME='train'

DATASET="coco"
BATCHSIZE=4
NUM_SAMPLES=50
NUM_ITER=500

# Define path
LOGDIR="./data"
DATADIR="" # change data path

# echo "Train SD1.5 start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
# python train.py --model sd1.5 --dataset $DATASET --batch_size $BATCHSIZE --separate_gpus \
#  --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
#  --n_dc_tokens 4 --apply_dc True False False --epochs 2 --dweight 10

# echo "Train SDXL start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
# python train.py --model sdxl --dataset $DATASET --batch_size $BATCHSIZE --separate_gpus \
#  --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
#  --n_dc_tokens 4 --apply_dc True True False --epochs 1 --dweight 10

echo "Train SD3 start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
python train.py --model sd3 --dataset $DATASET --batch_size $BATCHSIZE --separate_gpus \
 --datadir $DATADIR --logdir $LOGDIR --num_iter $NUM_ITER --num_eval $NUM_SAMPLES \
 --n_dc_tokens 4 --n_dc_layers 5 --epochs 2 --use_dc_t --dweight 0 

echo "All epochs completed!"