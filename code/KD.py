import tensorflow as tf
from tensorflow.keras import Model
import yaml
import pickle
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

from load_data import data_loader
from baseline_kd_loss import *

set_seeds()


class Distilling(Model):
    def __init__(self, student_model=None, teacher_model=None, train_mode='teacher-TA_1', loss_mode='fitnet', **kwargs):
        super(Distilling, self).__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_mode = train_mode
        self.loss_mode = loss_mode

        self.T = 0.
        self.alpha = 0.

        self.clf_loss = None
        self.kd_loss = None
        self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.kd_loss_tracker = tf.keras.metrics.Mean(name='kd_loss')
        self.baseline_loss_tracker = tf.keras.metrics.Mean(name='baseline_loss')
        self.sum_loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, clf_loss=None, kd_loss=None, T=0., alpha=0.,  **kwargs):
        super(Distilling, self).compile(**kwargs)
        self.clf_loss = clf_loss
        self.kd_loss = kd_loss
        self.T = T
        '''
            温度系数 T 在知识蒸馏中非常重要，通常用来控制 softmax 的平滑程度。
            温度越高，softmax 的输出分布越平滑，
            反之，温度越低，softmax 输出会更接近于 one-hot 分布。
            较高的温度帮助学生模型从教师模型中学习更丰富的信息，尤其是在不同类的相对概率之间的关系上。
        '''
        self.alpha = alpha

    @property
    def metrics(self):
        metrics = [self.sum_loss_tracker, self.clf_loss_tracker
                   , self.baseline_loss_tracker, self.kd_loss_tracker]

        if self.compiled_metrics is not None:
            metrics += self.compiled_metrics.metrics

        return metrics

    def train_step(self, data):
        x, y = data
        video_data, eeg_data = x

        with tf.GradientTape() as tape:
            if self.train_mode.endswith('student'):
                student_cls, student_logits, student_att = self.student_model(video_data)
            else:
                student_cls, student_logits = self.student_model(x)
            if self.train_mode.startswith('teacher'):
                teacher_cls, _, _, teacher_logits, teacher_att = self.teacher_model(x, training=False)
            else:
                teacher_cls, teacher_logits = self.teacher_model(x, training=False)

            clf_loss_value = self.clf_loss(y, student_cls)
            kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))
            if self.loss_mode == "fitnet":
                baseline_loss_value = tf.reduce_mean(tf.square(teacher_att - student_att))
            elif self.loss_mode == "NST":
                # 计算 MMD 损失
                gamma = 1.0 / (2.0 * tf.reduce_mean(tf.square(student_att)))  # 根据特征自动计算 gamma
                baseline_loss_value = mmd_loss(teacher_att, student_att, gamma=gamma)
            elif self.loss_mode == "DKD":
                temperature = 4.0
                alpha = 1.0
                beta = 1.0
                # 计算 DKD 损失
                baseline_loss_value = dkd_loss(teacher_logits, student_logits, y, alpha, beta, temperature)

            sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * baseline_loss_value

        self.optimizer.minimize(sum_loss_value, self.student_model.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.baseline_loss_tracker.update_state(baseline_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        video_data, eeg_data = x

        if self.train_mode.endswith('student'):
            student_cls, student_logits, student_att = self.student_model(video_data)
        else:
            student_cls, student_logits = self.student_model(x)
        if self.train_mode.startswith('teacher'):
            teacher_cls, _, _, teacher_logits, teacher_att = self.teacher_model(x, training=False)
        else:
            teacher_cls, teacher_logits = self.teacher_model(x, training=False)

        clf_loss_value = self.clf_loss(y, student_cls)
        kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))
        baseline_loss_value = tf.reduce_mean(tf.square(teacher_att - student_att))
        sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * baseline_loss_value

        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.baseline_loss_tracker.update_state(baseline_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}
    

def distilling_train(args):
    print('args: ',args)

    teacher_str, student_str = split_string(args.train_mode)

    # 加载数据集以及教师模型路径
    if args.dataset == "DEAP":
        data_path = f"./data/save_data/DEAP_{args.data_type}_Trans_final.pkl"
        teacher_model_path = f"./ts_models/{teacher_str}/{args.data_type}/best_model_1e-04"
    else:
        data_path = f"/root/autodl-tmp/HCI_data/HCI_{args.data_type}_4s.pkl"
        teacher_model_path = f"./ts_models/{teacher_str}/{args.data_type}/HCI_best_model_1e-04"
    train_dataset, val_dataset, test_dataset = data_loader(data_path, args.batch_size)

    # 提取 config 中的 learning_rate
    learning_rate_str = f"{args.learning_rate:.0e}"  # 将学习率格式化为科学计数法
    print('learning_rate_str',learning_rate_str)


    # args.train_mode == 'teacher-student':
    teacher_model = load_model(teacher_model_path)
    teacher_model.summary()
    student_model = VideoModel()


    # # 确定模型的保存路径
    # model_path = f'./ts_models/{student_str}/{args.data_type}/HCI_best_model_{learning_rate_str}'

    # print("model_path",model_path)
    # # 创建保存路径的文件夹（如果不存在）
    # if not os.path.exists(os.path.dirname(model_path)):
    #     print('路径不存在')
    #     os.makedirs(os.path.dirname(model_path))
    #     print('创建路径')


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



    dist = Distilling(student_model=student_model, teacher_model=teacher_model, train_mode=args.train_mode, loss_mode=args.loss_mode)
    dist.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True),
        clf_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), # 表示模型的输出已经是经过 softmax 的概率分布，因此不需要再应用 softmax。损失函数直接基于这些概率分布计算交叉熵。
        kd_loss=tf.keras.losses.KLDivergence(),
        T=2,
        alpha=args.alpha, # 用来调整clf_loss 和 kd_loss的配比
        metrics=[tf.keras.metrics.CategoricalAccuracy('acc')]
    )

    callback_list=[]
    # if args.save_model:
    #     # 使用自定义的回调，只保存学生模型
    #     save_best_student_callback = SaveBestStudentModelCallback(save_path=model_path, monitor='val_acc', mode='max')
    #     callback_list.append(save_best_student_callback)

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

    test_loss, _, _, _, test_acc = dist.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    outcome = dist.evaluate(test_dataset)
    print('outcome', outcome)

    record_experiment(args, fittedModel, test_loss, test_acc, csv_file='KD_baseline.csv')

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_mode', type=str, default="teacher-student")
    parser.add_argument('--data_type', type=str, default="V")
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--loss_mode', type=str, default="DKD")
    parser.add_argument('--dataset', type=str, default="")

    # 数据路径和模型路径都用“dataset”参数控制
    # 将loss_mode和dataset参数添加好

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # 加载配置
    args = parser_args()

    distilling_train(args)
