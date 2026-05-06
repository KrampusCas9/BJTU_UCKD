import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表

from os import path
import pickle
import random as python_random
from copy import deepcopy

import numpy as np
import numpy.random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import warnings
import os


def split_string(s):
    """
    将字符串按 '-' 分隔，并返回前后两部分。

    参数:
    s (str): 要分隔的字符串，格式为 '部分1-部分2'

    返回:
    tuple: 包含两个元素的元组，分别是分隔符 '-' 前后的两部分
    """
    parts = s.split('-')
    if len(parts) != 2:
        raise ValueError("输入的字符串格式不符合预期，需要包含一个 '-' 分隔符")
    
    return parts[0], parts[1]


import csv
import os

def record_experiment(args, fittedModel, test_loss, test_acc, csv_file):
    # 定义需要记录的实验参数
    params_to_record = vars(args)  # 将args对象转为字典
    history = fittedModel.history  # 获取训练过程中的历史信息
    
    # 获取最佳验证准确率所在的epoch
    best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    train_acc = max(history['acc'])  # 训练中的最高准确率
    val_acc = max(history['val_acc'])  # 验证中的最高准确率
    train_loss = history['loss'][best_epoch - 1]  # 最佳验证epoch时的训练损失
    val_loss = history['val_loss'][best_epoch - 1]  # 最佳验证epoch时的验证损失
    
    # 组装要写入CSV的信息
    experiment_data = {
        'Best Epoch': best_epoch,
        'Train Accuracy': train_acc,
        'Val Accuracy': val_acc,
        'Train Loss': train_loss,
        'Val Loss': val_loss,
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
    }
    
    # 将args中的实验参数和记录信息合并
    experiment_data.update(params_to_record)

    # CSV文件头
    headers = list(experiment_data.keys())
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.isfile(csv_file)

    if not file_exists:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    # 打开文件并写入数据（追加模式）

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        # 写入实验数据
        writer.writerow(experiment_data)
    
    print(f"实验记录已追加到文件: {csv_file}")


def save_experiment_results(args, history, test_loss, test_acc, file_path="experiment_results.txt"):
    """将实验配置和最优验证准确率相关结果保存到一个文本文件中"""
    
    # 找出验证准确率最高的 epoch
    best_epoch = max(range(len(history.history['val_acc'])), key=lambda i: history.history['val_acc'][i])
    best_val_acc = history.history['val_acc'][best_epoch]
    best_val_loss = history.history['val_loss'][best_epoch]
    best_train_acc = history.history['acc'][best_epoch]
    best_train_loss = history.history['loss'][best_epoch]
    
    # 将信息写入文件
    with open(file_path, "a") as f:  # "a" 模式表示追加写入
        f.write(f"Experiment with args:\n")
        f.write(f"PMR_loss: {args.PMR_loss}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epoch}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Data Type: {args.data_type}\n")
        f.write(f"Distance Type: {args.distance_type}\n")
        f.write(f"Alpha: {args.alpha}\n")
        
        f.write("\nBest Epoch Results:\n")
        f.write(f"  Best Epoch: {best_epoch + 1}\n")  # +1 是因为索引从 0 开始
        f.write(f"  Train Loss: {best_train_loss:.4f}, Train Accuracy: {best_train_acc:.4f}\n")
        f.write(f"  Val Loss: {best_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}\n")

        f.write("\nTest Results:\n")
        f.write(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
        f.write("\n" + "="*50 + "\n\n")  # 分隔符

def set_seeds(np_seed=1234, py_seed=1234, tf_seed=1234):
    np.random.seed(np_seed)
    python_random.seed(py_seed)
    tf.random.set_seed(tf_seed)

# def load_teacher(dataset, data_type, leak=False):
#     _, obj = get_MultiModal_model()
#     if leak:
#         final_path = f"/root/autodl-tmp/code/models/teachers/{dataset}/{data_type}/Trans_teacher_leak_final.h5"
#     else:
#         final_path = f"/root/autodl-tmp/code/models/teachers/{dataset}/{data_type}/Trans_teacher_final.h5"

#     teacher = load_model(final_path, custom_objects=obj)

#     return teacher

def Add_Gaussian_for_Channel(signal, SNR=5):
    Ps = np.sum(abs(signal) ** 2) / signal.shape[0]
    Pn = Ps / (10 ** ((SNR / 10)))
    noise = np.random.randn(signal.shape[0], 1) * np.sqrt(Pn)
    signal_noise = signal + noise
    return signal_noise


def Add_Gaussian(dataset, SNR=5):
    noised_dataset = deepcopy(dataset)

    for sample in tqdm(range(noised_dataset.shape[0])):
        for channel in range(noised_dataset.shape[1]):
            noised_dataset[sample, channel, :, :] = Add_Gaussian_for_Channel(noised_dataset[sample, channel, :, :])

    return noised_dataset


def load_Gaussian_data(dataset, data_type):
    with open(f"/root/autodl-tmp/code/data/save_data/{dataset}_{data_type}_aug_final.pkl", "rb") as f:
        data = pickle.load(f)

    train_set, valid_set, test_set = data["train"], data["valid"], data["test"]

    return train_set, valid_set, test_set



# 记录计算过的层
layer_outputs = {}

# 返回当前层的output
def get_output_of_layer(layer, starting_layer_name, new_input):

    # 1、如果已经计算过则直接返回output
    if layer.name in layer_outputs:
        return layer_outputs[layer.name]

    # 2、如果回溯到input节点则通过层计算output
    if layer.name == starting_layer_name:
        out = layer(new_input)
        layer_outputs[layer.name] = out
        return out

    # 3、找到当前层连接的所有输入层
    prev_layers = []
    for node in layer._inbound_nodes:
        prev_layers.extend([node.inbound_layers])

    # 递归得到所有前一层的output
    pl_outs = []
    for pl in prev_layers:
        pl_outs.extend([get_output_of_layer(pl, starting_layer_name, new_input)])

    # 通过得到的output计算当前层output
    out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
    layer_outputs[layer.name] = out
    return out


if __name__ == "__main__":
    # test_array = numpy.random.random((128, 1))
    # Add_Gaussian(test_array)
    #generate_data("HCI", "V")
    print()