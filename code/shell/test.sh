#!/bin/bash

# 定义参数变量
PMR_LOSS=False
BATCH_SIZE=128
EPOCH=100
LEARNING_RATE=1e-4
DATA_TYPE="A"
DISTANCE_TYPE="cosine"
ALPHA=2
SAVE_MODEL=False
CONTRASTIVE_LEARNING=True

# 运行 Python 脚本并传递参数
python ./code/train_teacher.py \
    --PMR_loss $PMR_LOSS \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCH \
    --learning_rate $LEARNING_RATE \
    --data_type $DATA_TYPE \
    --distance_type $DISTANCE_TYPE \
    --alpha $ALPHA \
    --save_model $SAVE_MODEL \
    --contrastive_learning $CONTRASTIVE_LEARNING
