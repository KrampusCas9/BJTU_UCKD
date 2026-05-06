import tensorflow as tf

def mmd_loss(teacher_features, student_features, gamma=1.0):
    """
    使用高斯核计算 MMD 损失。
    teacher_features: 教师网络的特征张量 [N, D]
    student_features: 学生网络的特征张量 [N, D]
    gamma: 高斯核参数，默认值为 1.0
    """
    def gaussian_kernel(x, y, gamma):
        """
        计算高斯核矩阵。
        x: 特征张量 [N, D]
        y: 特征张量 [M, D]
        gamma: 高斯核的超参数
        """
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)  # [N, 1]
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)  # [M, 1]
        cross_term = tf.matmul(x, y, transpose_b=True)  # [N, M]
        distances = x_norm - 2 * cross_term + tf.transpose(y_norm)  # [N, M]
        return tf.exp(-gamma * distances)

    # 计算核矩阵
    k_xx = gaussian_kernel(teacher_features, teacher_features, gamma)
    k_yy = gaussian_kernel(student_features, student_features, gamma)
    k_xy = gaussian_kernel(teacher_features, student_features, gamma)

    # 计算 MMD 损失
    mmd = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)
    return mmd

def dkd_loss(logits_teacher, logits_student, labels, alpha=1.0, beta=1.0, temperature=4.0):
    # 转换 labels 的数据类型为 float32
    labels = tf.cast(labels, tf.float32)

    # 计算 softmax 概率分布（加温度平滑）
    probs_teacher = tf.nn.softmax(logits_teacher / temperature, axis=-1)
    probs_student = tf.nn.softmax(logits_student / temperature, axis=-1)

    # 提取目标类别的概率
    target_probs_teacher = tf.reduce_sum(probs_teacher * labels, axis=1, keepdims=True)  # [batch_size, 1]
    target_probs_student = tf.reduce_sum(probs_student * labels, axis=1, keepdims=True)  # [batch_size, 1]

    # 非目标类别的概率分布
    non_target_probs_teacher = probs_teacher * (1 - labels)
    non_target_probs_teacher /= tf.reduce_sum(non_target_probs_teacher, axis=1, keepdims=True)  # 归一化
    non_target_probs_student = probs_student * (1 - labels)
    non_target_probs_student /= tf.reduce_sum(non_target_probs_student, axis=1, keepdims=True)  # 归一化

    # 计算 TCKD 损失（目标类别的二分类 KL 散度）
    tckd_loss = tf.reduce_mean(
        tf.keras.losses.KLDivergence()(
            tf.stop_gradient(target_probs_teacher), target_probs_student
        )
    )

    # 计算 NCKD 损失（非目标类别的 KL 散度）
    nckd_loss = tf.reduce_mean(
        tf.keras.losses.KLDivergence()(
            tf.stop_gradient(non_target_probs_teacher), non_target_probs_student
        )
    )

    # 组合损失
    dkd_loss = alpha * tckd_loss + beta * nckd_loss

    return dkd_loss

