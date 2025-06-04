#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
NODE='node-name'
NAME='train'

DATASET="coco"
BATCHSIZE=8
NROWS=4
NUM_EVAL=50
NUMITER=500

# Define path
LOGDIR="./data"
DATADIR="" # change data path

# echo "Train SD1.5 start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
# torchrun --nproc-per-node=2 train_dist.py --model sd1.5 --datadir $DATADIR --logdir $LOGDIR \
#  --dataset $DATASET --batch_size $BATCHSIZE --epochs 2 --num_iter $NUMITER --num_eval $NUM_EVAL --nrows $NROWS \
#  --n_dc_tokens 4 --apply_dc True False False --use_dc_t --dweight 10 \

# echo "Train SDXL start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
# torchrun --nproc-per-node=2 train_dist.py --model sdxl --datadir $DATADIR --logdir $LOGDIR \
#  --dataset $DATASET --batch_size $BATCHSIZE --epochs 2 --num_iter $NUMITER --num_eval $NUM_EVAL --nrows $NROWS \
#  --n_dc_tokens 4 --apply_dc True False False --dweight 10 \

echo "Train SD3 start"
# srun -w $NODE -N 1 -n 1 --cpus-per-task=2 --job-name=$NAME \ #only for slurm
torchrun --nproc-per-node=2 train_dist.py --model sd3 --datadir $DATADIR --logdir $LOGDIR \
 --dataset $DATASET --batch_size $BATCHSIZE --epochs 2 --num_iter $NUMITER --num_eval $NUM_EVAL --nrows $NROWS \
 --n_dc_tokens 4 --n_dc_layers 5 --use_dc_t --dweight 0 \

echo "All epochs completed!"