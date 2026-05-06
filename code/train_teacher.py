import tensorflow as tf
from tensorflow.keras import optimizers, metrics, callbacks
import yaml
import pickle
import tqdm
from model import TS_MultiModalModel, PMRCallback
from utils import set_seeds, save_experiment_results, record_experiment
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from load_data import data_loader

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--data_type', type=str, default="V")
    parser.add_argument('--data_name', type=str, default="DEAP")

    parser.add_argument('--PMR_loss', type=bool, default=True, help="is use the PMR_loss? (default: false)")
    parser.add_argument('--distance_type', type=str, default="euclidean")
    parser.add_argument('--alpha', type=float, default=1)

    parser.add_argument('--contrastive_learning', type=bool ,default=False)

    parser.add_argument('--save_model', type=bool, default=True)
    

    args = parser.parse_args()

    return args

set_seeds()

# 加载配置信息
args = parser_args()
print('args:', args)

# 加载数据集
Dataset_Name=args.data_name
if Dataset_Name == "HCI":
    data_path = f"./data/save_data/HCI_{args.data_type}_4s.pkl"
elif Dataset_Name == "DEAP":
    data_path = f"./data/save_data/DEAP_{args.data_type}_Trans_final.pkl"
train_dataset, val_dataset, test_dataset = data_loader(data_path, args.batch_size)

# 初始化模型
model = TS_MultiModalModel(args)

# # 构建模型并查看其架构
# model.build(input_shape=[(10, 5, 768, 1),(10, 28, 512, 1)])
# model.summary()

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True)
)


# 确定模型的保存路径

# 提取 config 中的 learning_rate
learning_rate_str = f"{args.learning_rate:.0e}"  # 将学习率格式化为科学计数法
print('learning_rate_str',learning_rate_str)

if args.PMR_loss:
    model_path = f'ts_models/{Dataset_Name}/teacher/{args.data_type}/PMR_model_{learning_rate_str}'
else:
    model_path = f'ts_models/{Dataset_Name}/teacher/{args.data_type}/wo_PMR_model_{learning_rate_str}'

print("model_path",model_path)
# 创建保存路径的文件夹（如果不存在）
if not os.path.exists(os.path.dirname(model_path)):
    print('路径不存在')
    os.makedirs(os.path.dirname(model_path))
    print('创建路径')


# 定义保存模型的回调
save_model_callback = ModelCheckpoint(
    filepath=model_path,  # 保存模型的路径
    save_weights_only=False,  # 保存完整的模型，不仅仅是权重
    monitor='val_acc',  # 监控验证集的准确率
    mode='max',  # 取最高值保存
    save_best_only=True  # 仅保存验证集准确率最高的模型
)

# 定义更新beta、gamma参数的回调
PMR_callback = PMRCallback(model)

# 定义早停回调
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',       # 监控的指标（可以改为 'val_accuracy' 或其他指标）
    patience=20,               # 若干轮内无改善则停止
    restore_best_weights=True # 恢复至最佳权重
)

callback_list = []
callback_list.append(early_stopping)

if args.PMR_loss:
    callback_list.append(PMR_callback)
if args.save_model:
    callback_list.append(save_model_callback)

# 训练模型，并在验证集准确率最高时保存模型
fittedModel = model.fit(
    train_dataset,
    epochs=args.epoch,
    validation_data=val_dataset,
    verbose=1,
    callbacks=callback_list  # 添加模型保存回调
)

# 获取验证集最高的准确率及其索引
best_val_accuracy = max(fittedModel.history['val_acc'])
best_epoch_index = fittedModel.history['val_acc'].index(best_val_accuracy)

# 获取对应的训练准确率
best_train_accuracy = fittedModel.history['acc'][best_epoch_index]

# 打印验证集最高准确率和对应的训练准确率，保留4位小数
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Training accuracy at best validation accuracy: {best_train_accuracy:.4f}")

test_loss, test_acc = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# 定义度量对象来计算准确率
test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# 遍历验证数据集，计算教师模型的准确率
for x_batch, y_batch in test_dataset:
    # 获取视频和EEG数据
    video_data, eeg_data = x_batch
    
    # 前向传播，得到模型预测
    teacher_cls = model((video_data, eeg_data), training=False)[0]
    
    # 更新准确率度量对象
    test_accuracy_metric.update_state(y_batch, teacher_cls)

# 获取最终的准确率
final_accuracy = test_accuracy_metric.result().numpy()
print(f"Teacher Model Accuracy on test Set: {final_accuracy:.4f}")


record_experiment(args, fittedModel, test_loss, test_acc, csv_file='./result_csv/train_teacher_result.csv')