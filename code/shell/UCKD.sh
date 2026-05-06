#!/bin/bash

# 设置一些通用参数
data_types=("A" "V")
ALPHA=0.5
SAVE_MODEL=True
EPOCH=120
DATA_NAME="HCI"
learning_rate=1e-5


# declare -a learning_rates=(1e-4)
# declare -a train_modes=("teacher-TA_1" "TA_1-TA_2" "TA_2-student")

declare -a learning_rates=(5e-5 1e-4 5e-4 1e-3)
declare -a train_modes=("TA_2-student")

for dt in "${data_types[@]}"; do
    for train_mode in "${train_modes[@]}"; do
        for lr in "${learning_rates[@]}"; do
            # 运行 Python 脚本并传递参数
            python ./code/train_UCKD.py \
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
