import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
import os

# 获取当前工作目录
current_path = os.getcwd()
print(f"当前工作路径为: {current_path}")

# 获取父文件夹（二级）的路径
level_two_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(level_two_dir)
print(level_two_dir)

from load_data import data_loader



# ------------------------------- #
# 1. 配置与初始化
# ------------------------------- #

# 数据和模型路径配置
data_type = "A"
teacher_model_path = f"./ts_models/teacher/{data_type}/wo_PMR_model_1e-04"
data_path = f"./data/save_data/DEAP_{data_type}_Trans_final.pkl"

# 输出路径
output_dir = "./feature_visualizations/"
os.makedirs(output_dir, exist_ok=True)

# 加载教师模型
model = load_model(teacher_model_path)
model.summary()

# 加载数据集
train_dataset, val_dataset, test_dataset = data_loader(data_path, 128)

# ------------------------------- #
# 2. 可视化函数
# ------------------------------- #


# 定义函数进行降维并按标签可视化
def visualize_features_with_labels(video_features, eeg_features, labels, save_path, method="TSNE"):
    """
    可视化视频和EEG特征，支持不同降维方法。

    参数：
    - video_features: 视频模态特征 (NumPy array)
    - eeg_features: EEG模态特征 (NumPy array)
    - labels: 数据标签 (one-hot 编码)
    - save_path: 保存图片的路径
    - method: 降维方法 ("PCA" 或 "TSNE")
    """
    # 降维
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "TSNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    # 降维到2D
    video_reduced = reducer.fit_transform(video_features)
    eeg_reduced = reducer.fit_transform(eeg_features)
    
    # 将标签转换为 NumPy 数组
    labels = np.argmax(labels, axis=1)  # 假设 labels 是 one-hot 编码

    # 绘制
    plt.figure(figsize=(8, 8))
    colors = ['red', 'blue', 'green', 'orange']  # 不同类别的颜色
    for label in np.unique(labels):
        plt.scatter(
            video_reduced[labels == label, 0],
            video_reduced[labels == label, 1],
            label=f'Video Class {label}', alpha=0.5, marker='o'
        )
        plt.scatter(
            eeg_reduced[labels == label, 0],
            eeg_reduced[labels == label, 1],
            label=f'EEG Class {label}', alpha=0.5, marker='x'
        )
    
    # 添加图例和标题
    plt.legend()
    plt.title(f"Video and EEG Features (Colored by Labels, {method})")
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()

# ------------------------------- #
# 3. 特征提取与评估
# ------------------------------- #

# 定义度量对象
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# 用于收集特征和标签的列表
video_features_list = []
eeg_features_list = []
flatten_features_list = []
labels_list = []

# 遍历验证数据集，逐批次处理
for x_batch, y_batch in val_dataset:
    # 分离视频和EEG数据
    video_data, eeg_data = x_batch

    # 获取分类结果和中间特征
    cls, video_features, eeg_features, logit_feature, flatten_features = model(
        (video_data, eeg_data), training=False
    )
    
    # 更新准确率度量
    accuracy_metric.update_state(y_batch, cls)
    
    # 收集中间特征和标签
    video_features_list.append(video_features.numpy())
    eeg_features_list.append(eeg_features.numpy())
    flatten_features_list.append(flatten_features.numpy())
    labels_list.append(y_batch.numpy())

# 获取最终准确率
val_accuracy = accuracy_metric.result().numpy()
print(f"Teacher Model Accuracy on Validation Set: {val_accuracy:.4f}")

# 将收集到的特征和标签拼接为完整数组
video_features = np.concatenate(video_features_list, axis=0)
eeg_features = np.concatenate(eeg_features_list, axis=0)
flatten_features = np.concatenate(flatten_features_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

# ------------------------------- #
# 4. 可视化与保存
# ------------------------------- #

# 可视化单独模态特征
visualize_features_with_labels(
    video_features, eeg_features, labels, 
    save_path=os.path.join(output_dir, "scatter.png")
)

# # 保存特征为 .npy 文件
# np.save('video_features.npy', video_features)
# np.save('eeg_features.npy', eeg_features)
# np.save('flatten_features.npy', flatten_features)
# np.save('labels.npy', labels)

print(f"Feature visualizations saved in {output_dir}")
