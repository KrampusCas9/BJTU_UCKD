import tensorflow as tf
import yaml
import pickle
import wandb
import tqdm
import os
import argparse
import csv

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import clone_model, load_model

from utils import set_seeds, split_string, record_experiment
from StudentModel import VideoModel, EEGModel
from TAModel import TA_MultiModalModel_1, TA_MultiModalModel_2

from wandb.integration.keras import WandbCallback

from distill import Distilling, SaveBestStudentModelCallback
from load_data import data_loader

set_seeds()

def distilling_train(args):
    print('args: ',args)
    Dataset_Name = args.data_name

    # 加载数据集
    if Dataset_Name == "HCI":
        data_path = f"./data/save_data/HCI_{args.data_type}_4s.pkl"
    elif Dataset_Name == "DEAP":
        data_path = f"./data/save_data/DEAP_{args.data_type}_Trans_final.pkl"
    train_dataset, val_dataset, test_dataset = data_loader(data_path, args.batch_size)

    # cosine_lr_schdule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=config['lr_schedule']['initial_lr'],
    #     decay_steps=config['lr_schedule']['decay_step'],
    #     alpha=config['lr_schedule']['alpha'],
    # )



    # 提取 config 中的 learning_rate
    learning_rate_str = f"{args.learning_rate:.0e}"  # 将学习率格式化为科学计数法
    print('learning_rate_str',learning_rate_str)

    teacher_str, student_str = split_string(args.train_mode)
    if args.train_mode == 'teacher-TA_1':
        # teacher_model_path = f"./ts_models/{Dataset_Name}/{teacher_str}/{args.data_type}/{Dataset_Name}_best_model_1e-04"
        teacher_model_path = f"./ts_models/{Dataset_Name}/{teacher_str}/{args.data_type}/wo_PMR_model_1e-04"
        teacher_model = load_model(teacher_model_path)
        teacher_model.summary()
        student_model = TA_MultiModalModel_1()
    elif args.train_mode == 'teacher-student':
        teacher_model_path = f"./ts_models/{Dataset_Name}/{teacher_str}/{args.data_type}/{Dataset_Name}_best_model_1e-04"
        teacher_model = load_model(teacher_model_path)
        teacher_model.summary()
        student_model = VideoModel()
    else:
        teacher_model_path = f'./ts_models/{Dataset_Name}/{teacher_str}/{args.data_type}/{Dataset_Name}_best_model_{learning_rate_str}'
        teacher_model = load_model(teacher_model_path)
        if args.train_mode == 'TA_1-TA_2':
            student_model = TA_MultiModalModel_2()
        else:
            student_model = VideoModel()


    # 确定模型的保存路径
    model_path = f'./ts_models/{Dataset_Name}/{student_str}/{args.data_type}/{Dataset_Name}_best_model_{learning_rate_str}'

    print("model_path",model_path)
    # 创建保存路径的文件夹（如果不存在）
    if not os.path.exists(os.path.dirname(model_path)):
        print('路径不存在')
        os.makedirs(os.path.dirname(model_path))
        print('创建路径')


    # 定义度量对象来计算准确率
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    # 遍历验证数据集，计算教师模型的准确率
    for x_batch, y_batch in val_dataset:
        # 获取视频和EEG数据
        video_data, eeg_data = x_batch
        
        # 前向传播，得到模型预测
        teacher_cls = teacher_model((video_data, eeg_data), training=False)[0]
        
        # 更新准确率度量对象
        accuracy_metric.update_state(y_batch, teacher_cls)

    # 获取最终的准确率
    val_accuracy = accuracy_metric.result().numpy()
    print(f"Teacher Model Accuracy on Validation Set: {val_accuracy:.4f}")


    # 定义度量对象来计算准确率
    test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    # 遍历验证数据集，计算教师模型的准确率
    for x_batch, y_batch in test_dataset:
        # 获取视频和EEG数据
        video_data, eeg_data = x_batch
        
        # 前向传播，得到模型预测
        teacher_cls = teacher_model((video_data, eeg_data), training=False)[0]
        
        # 更新准确率度量对象
        test_accuracy_metric.update_state(y_batch, teacher_cls)

    # 获取最终的准确率
    test_accuracy = test_accuracy_metric.result().numpy()
    print(f"Teacher Model Accuracy on test Set: {test_accuracy:.4f}")


    # 要保存的CSV文件路径
    csv_file_path = 'self_csv.csv'
    # 检查CSV文件是否存在
    file_exists = os.path.isfile(csv_file_path)
    # 以追加模式打开文件，若文件不存在则会自动创建
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 如果文件是新创建的，写入表头
        if not file_exists:
            writer.writerow(["Teacher Model Path", "Validation Accuracy", "Test Accuracy"])
        
        # 写入教师模型路径及其验证集和测试集的准确率
        writer.writerow([teacher_model_path, f"{val_accuracy:.4f}", f"{test_accuracy:.4f}"])

    print(f"Data has been written to {csv_file_path}")



    dist = Distilling(student_model=student_model, teacher_model=teacher_model, train_mode=args.train_mode)
    dist.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True),
        clf_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # 表示模型的输出已经是经过 softmax 的概率分布，因此不需要再应用 softmax。损失函数直接基于这些概率分布计算交叉熵。
        kd_loss=tf.keras.losses.KLDivergence(),
        T=2,
        alpha=args.alpha, # 用来调整clf_loss 和 kd_loss的配比
        metrics=[tf.keras.metrics.CategoricalAccuracy('acc')]
    )

    callback_list=[]
    if args.save_model:
        # 使用自定义的回调，只保存学生模型
        save_best_student_callback = SaveBestStudentModelCallback(save_path=model_path, monitor='val_acc', mode='max')
        callback_list.append(save_best_student_callback)

    fittedModel = dist.fit(x=train_dataset
                           , epochs=args.epoch
                           , callbacks=callback_list
                           , validation_data=val_dataset
                           , verbose=1
                           )

    # 获取验证集最高的准确率及其索引
    best_val_accuracy = max(fittedModel.history['val_acc'])
    best_epoch_index = fittedModel.history['val_acc'].index(best_val_accuracy)

    # 获取对应的训练准确率
    best_train_accuracy = fittedModel.history['acc'][best_epoch_index]

    # 打印验证集最高准确率和对应的训练准确率，保留4位小数
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Training accuracy at best validation accuracy: {best_train_accuracy:.4f}")

    dist.evaluate(train_dataset)
    dist.evaluate(val_dataset)

    test_loss, _, _, test_acc = dist.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    outcome = dist.evaluate(test_dataset)
    print('outcome', outcome)

    record_experiment(args, fittedModel, test_loss, test_acc, csv_file='TA_result.csv')

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_mode', type=str, default="TA_1-TA_2",help=["TA_1-TA_2", "teacher-TA_1", "teacher-student"])
    parser.add_argument('--data_type', type=str, default="A")
    parser.add_argument('--data_name', type=str, default="DEAP")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--save_model', type=bool, default=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # 加载配置
    args = parser_args()


    # # 2. 初始化 WandB
    # wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'], config=config)

    distilling_train(args)
