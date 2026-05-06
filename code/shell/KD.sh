#!/bin/bash

# 设置一些通用参数
DATA_TYPE="V"
ALPHA=0.5
SAVE_MODEL=True
EPOCH=100
learning_rate=1e-4

# declare -a learning_rates=(1e-5 5e-5 1e-4 5e-4)
declare -a train_modes=("teacher-student")

for train_mode in "${train_modes[@]}"; do
    # 运行 Python 脚本并传递参数
    python ./code/train_TA.py \
        --epoch $EPOCH \
        --learning_rate $learning_rate \
        --train_mode $train_mode \
        --data_type $DATA_TYPE \
        --alpha $ALPHA \
        --save_model $SAVE_MODEL
done
