#!/bin/bash

# 设置一些通用参数
data_types=("A" "V")
DATA_NAME="DEAP"
ALPHA=0.5
SAVE_MODEL=True
EPOCH=100


declare -a learning_rates=(1e-5 5e-5 1e-4 5e-4)
declare -a train_modes=("teacher-TA_1" "TA_1-TA_2")

for lr in "${learning_rates[@]}"; do
    for dt in "${data_types[@]}"; do
        for train_mode in "${train_modes[@]}"; do
            python ./code/train_TA.py \
            --epoch $EPOCH \
            --learning_rate $lr \
            --train_mode $train_mode \
            --data_type $dt \
            --data_name $DATA_NAME \
            --alpha $ALPHA \
            --save_model $SAVE_MODEL
        done
    done
done
