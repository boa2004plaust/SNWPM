#!/bin/bash
# Build documentation for display in web browser.

LOG=log
ARCH=SNWPM_FAAM
MODE=orig
NAME=using_orig
EPOCH=200
BS=100
LR=0.01


MODE=aug12
NAME=using_$MODE
LOG=log_$MODE
CUDA_VISIBLE_DEVICES=0 python train.py \
                        --exp-dir $LOG \
                        --arch $ARCH \
                        --epoch $EPOCH \
                        --lr $LR \
                        --bs $BS \
                        --name $NAME \
                        --mode $MODE