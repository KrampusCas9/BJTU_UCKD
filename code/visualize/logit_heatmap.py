import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
import os

import seaborn as sns

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
data_type = "V"
teacher_model_path = f"./ts_models/teacher/{data_type}/best_model_1e-04"
wo_TA_student_model_path = f""
student_model_path = f"/root/autodl-tmp/my_EmotionKD/ts_models/student/{data_type}/DGKD_best_model_1e-04"
data_path = f"./data/save_data/DEAP_{data_type}_Trans_final.pkl"

# 输出路径
output_dir = "./logits_heatmap_visualizations/"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
teacher_model = load_model(teacher_model_path)
student_model = load_model(student_model_path)

# 加载数据集
train_dataset, val_dataset, test_dataset = data_loader(data_path, 128)
student_train_dataset, student_val_dataset, student_test_dataset = data_loader(data_path, 128, only_video=True)

# ------------------------------- #
# 2. 可视化函数
# ------------------------------- #


def generate_and_save_heatmap(teacher_logits, student_logits, save_path, title="Difference of Correlation Matrices"):
    """
    生成并保存教师和学生网络logits相关性矩阵的差异热力图。
    
    参数：
    - teacher_logits: 教师网络的logits (NumPy array, shape: [N, C])
    - student_logits: 学生网络的logits (NumPy array, shape: [N, C])
    - save_path: 热力图保存路径
    - title: 热力图标题 (默认: "Difference of Correlation Matrices")
    """
    def compute_correlation_matrix(logits):
        """
        计算logits的相关性矩阵。
        :param logits: (N, C) 矩阵，其中N是样本数，C是类别数。
        :return: (C, C) 的相关性矩阵。
        """
        # 计算相关系数矩阵
        return np.corrcoef(logits.T)

    # 计算教师和学生的相关性矩阵
    teacher_corr = compute_correlation_matrix(teacher_logits)
    student_corr = compute_correlation_matrix(student_logits)
    
    # 计算差异矩阵
    difference_matrix = teacher_corr - student_corr

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(difference_matrix, cmap='coolwarm', annot=False, fmt=".2f", cbar=True, vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Class")
    
    # 保存热力图
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap saved at: {save_path}")

from sklearn.metrics.pairwise import cosine_similarity

def plot_cosine_similarity(teacher_logits, student_logits, save_path):
    """
    计算并绘制教师和学生logits之间的余弦相似度分布。
    :param teacher_logits: 教师网络的logits (N, C)
    :param student_logits: 学生网络的logits (N, C)
    :param save_path: 保存图片路径
    """
    # 计算余弦相似度
    cos_sim = cosine_similarity(teacher_logits, student_logits).diagonal()
    
    plt.figure(figsize=(8, 6))
    plt.hist(cos_sim, bins=20, alpha=0.7, color='blue', density=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Cosine Similarity Distribution: Teacher vs Student")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Cosine similarity plot saved at: {save_path}")


def plot_tsne(teacher_logits, student_logits, save_path):
    """
    使用t-SNE降维并绘制教师和学生logits的2D可视化。
    :param teacher_logits: 教师网络的logits (N, C)
    :param student_logits: 学生网络的logits (N, C)
    :param save_path: 保存图片路径
    """
    tsne = TSNE(n_components=2, random_state=42)
    logits_combined = np.vstack([teacher_logits, student_logits])
    tsne_result = tsne.fit_transform(logits_combined)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(
        tsne_result[:len(teacher_logits), 0],
        tsne_result[:len(teacher_logits), 1],
        alpha=0.7, label='Teacher Logits'
    )
    plt.scatter(
        tsne_result[len(teacher_logits):, 0],
        tsne_result[len(teacher_logits):, 1],
        alpha=0.7, label='Student Logits'
    )
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE Visualization of Teacher and Student Logits")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved at: {save_path}")

# ------------------------------- #
# 3. 特征提取与评估
# ------------------------------- #

# 定义度量对象
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# 用于收集logits的列表
teacher_logits_list = []

# 遍历验证数据集，逐批次处理
for x_batch, y_batch in val_dataset:
    # 分离视频和EEG数据
    video_data, eeg_data = x_batch

    # 获取分类结果和中间特征
    cls, video_features, eeg_features, logit_feature, flatten_features = teacher_model(
        (video_data, eeg_data), training=False
    )
    
    # 更新准确率度量
    accuracy_metric.update_state(y_batch, cls)
    
    # 收集中间特征和标签
    teacher_logits_list.append(logit_feature.numpy())

# 获取最终准确率
val_accuracy = accuracy_metric.result().numpy()
print(f"Teacher Model Accuracy on Validation Set: {val_accuracy:.4f}")

# 将收集到的特征和标签拼接为完整数组
teacher_logits = np.concatenate(teacher_logits_list, axis=0)

# 定义度量对象
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# 用于收集logits的列表
student_logits_list = []

# 遍历验证数据集，逐批次处理
for x_batch, y_batch in student_val_dataset:

    # 获取分类结果和中间特征
    cls, logit_feature = student_model(
        x_batch, training=False
    )
    
    # 更新准确率度量
    accuracy_metric.update_state(y_batch, cls)
    
    # 收集中间特征和标签
    student_logits_list.append(logit_feature.numpy())

# 获取最终准确率
val_accuracy = accuracy_metric.result().numpy()
print(f"Student Model Accuracy on Validation Set: {val_accuracy:.4f}")

# 将收集到的特征和标签拼接为完整数组
student_logits = np.concatenate(student_logits_list, axis=0)

# ------------------------------- #
# 4. 可视化与保存
# ------------------------------- #

# 可视化单独模态特征
generate_and_save_heatmap(
    teacher_logits, student_logits, 
    save_path=os.path.join(output_dir, "heatmap.png")
)

plot_cosine_similarity(teacher_logits, student_logits, save_path=os.path.join(output_dir, "cosine_similarity.png"))

plot_tsne(teacher_logits, student_logits, save_path=os.path.join(output_dir, "tsne_logits.png") )


# # 保存特征为 .npy 文件
# np.save('video_features.npy', video_features)
# np.save('eeg_features.npy', eeg_features)
# np.save('flatten_features.npy', flatten_features)
# np.save('labels.npy', labels)

print(f"Feature visualizations saved in {output_dir}")
