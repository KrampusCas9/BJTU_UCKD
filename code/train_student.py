import tensorflow as tf
from tensorflow.keras import optimizers, metrics, callbacks
import wandb
import yaml
import pickle
import tqdm
from StudentModel import VideoModel
from utils import set_seeds, record_experiment
from wandb.integration.keras import WandbCallback
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from load_data import data_loader

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--data_type', type=str, default="A")
    parser.add_argument('--data_name', type=str, default="HCI")

    parser.add_argument('--save_model', type=bool, default=True)
    

    args = parser.parse_args()

    return args

set_seeds()

# 加载配置信息
args = parser_args()

Dataset_Name=args.data_name
# 加载数据集
if Dataset_Name == "HCI":
    data_path = f"./data/save_data/HCI_{args.data_type}_4s.pkl"
elif Dataset_Name == "DEAP":
    data_path = f"./data/save_data/DEAP_{args.data_type}_Trans_final.pkl"
train_dataset, val_dataset, test_dataset = data_loader(data_path, args.batch_size, only_video=True)

# 初始化 WandB
# wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'], config=config)


# 初始化模型
model = VideoModel()

# # 构建模型并查看其架构
# model.build(input_shape=[(10, 5, 768, 1)])
# model.summary()

# cosine_lr_schdule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=config['lr_schedule']['initial_lr'],
#     decay_steps=config['lr_schedule']['decay_step'],
#     alpha=config['lr_schedule']['alpha'],
# )

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, amsgrad=True)
)


callback_list = []

if args.save_model:

    # 确定模型的保存路径

    # 提取 config 中的 learning_rate
    learning_rate_str = f"{args.learning_rate:.0e}"  # 将学习率格式化为科学计数法
    print('learning_rate_str',learning_rate_str)

    model_path = f'ts_models/{Dataset_Name}/only_student/{args.data_type}/best_model_{learning_rate_str}'

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

# 结束WandB记录
# wandb.finish()

# 保存实验结果
# save_experiment_results(args, fittedModel, test_loss, test_acc, file_path="experiment_results.txt")
record_experiment(args, fittedModel, test_loss, test_acc, csv_file='student_result.csv')