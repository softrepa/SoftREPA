#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
NODE='node-name'

MODEL='sd3'
NFE=28
CFG=4.0
USEDCT=True
NTOKENS=4
NLAYERS=5
APPLYDC=""
IMGSIZE=1024
LOADDIR="tokens/sd3"

# MODEL='sd1.5'
# NFE=30
# CFG=7.0
# USEDCT=True
# NTOKENS=4
# NLAYERS=0
# APPLYDC="True False False"
# IMGSIZE=512
# LOADDIR="tokens/sd1.5"

# MODEL='sdxl'
# NFE=30
# CFG=7.0
# USEDCT=False
# NTOKENS=8
# NLAYERS=0
# APPLYDC="True True False"
# IMGSIZE=1024
# LOADDIR="tokens/sdxl"

# Define common parameters
DATASET="coco"
BENCHMARKS="ImageReward-v1.0,CLIP,PickScore,HPS"
NUM_SAMPLES=-1 # -1 for all samples, or specify a number
SAVEDIR="" #change save path
DATADIR="" # change data path
BATCHSIZE=4 #change batch size for sampling

echo "running sample.py.."
# srun -w $NODE \ #only for slurm
python sample.py --cfg_scale $CFG --NFE $NFE --model $MODEL --img_size $IMGSIZE --batch_size $BATCHSIZE \
    --use_dc --use_dc_t $USEDCT --n_dc_tokens $NTOKENS --n_dc_layers $NLAYERS --apply_dc $APPLYDC \
    --load_dir $LOADDIR --save_dir $SAVEDIR --datadir $DATADIR --num $NUM_SAMPLES --dataset $DATASET \

echo "running eval.py.."
# srun -w $NODE \ #only for slurm
python eval.py --load_dir $SAVEDIR --datadir $DATADIR\
    --load_name "${DATASET}-cfg${CFG}-dcTrue-dct${USEDCT}-nfe${NFE}" \
    --benchmark $BENCHMARKS \
    --num $NUM_SAMPLES

echo "Evaluation Completed!"