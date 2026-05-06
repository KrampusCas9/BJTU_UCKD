import tensorflow as tf

import tensorflow as tf

def compute_prototypes(z, labels, num_classes=2):
    """
        计算每个类别的原型向量，适用于独热编码的 labels
        并返回每个类的特征子集
    """
    # 将独热编码标签转换为类索引
    class_indices = tf.argmax(labels, axis=-1)  # (batch_size,)
    
    prototypes = []
    z_classes = []
    for class_id in range(num_classes):
        # 找出属于当前 class_id 的样本
        mask = tf.equal(class_indices, class_id)
        z_class = tf.boolean_mask(z, mask)  # 选择属于该类的特征
        # print(f"z_class.shape: {z_class.shape}")


        # 使用 tf.cond 判断 z_class 是否为空
        class_prototype = tf.cond(
            tf.equal(tf.size(z_class), 0),  # 如果 z_class 大小为 0
            lambda: tf.zeros_like(z[0]),    # 使用零向量作为原型
            lambda: tf.reduce_mean(z_class, axis=0)  # 否则计算均值作为原型
        )
        prototypes.append(class_prototype)
        z_classes.append(z_class)

    return tf.stack(prototypes), z_classes


def distance(z, c, distance_type='euclidean'):
    """计算特征 z 与类原型 c 之间的距离"""
    if distance_type == 'euclidean':
        return tf.reduce_mean(tf.square(z - c), axis=[0,1])  # 欧氏距离
    elif distance_type == 'cosine':
        z_norm = tf.nn.l2_normalize(z, axis=-1)
        c_norm = tf.nn.l2_normalize(c, axis=-1)
        return 1 - tf.reduce_mean(z_norm * c_norm, axis=-1)  # 余弦距离
    else:
        raise ValueError('Unsupported distance type')

def prototype_cross_entropy_loss(z, prototypes, labels, num_classes=2, distance_type='euclidean'):
    """计算原型交叉熵损失 L_PCE"""
    
    # 计算 z 到每个类原型的距离

    # 确保 labels 是整数类型
    labels = tf.cast(labels, tf.int32)
    distances = [distance(z[i], prototypes[i], distance_type) for i in range(num_classes)]  
    distances = tf.convert_to_tensor(distances)
    # print(f'distance: {distances.shape}')
    # 计算 log-softmax
    logits = -distances
    log_prob = (-1) * tf.nn.log_softmax(logits, axis=-1)
    # print(f"log_prob: {log_prob}")
    
    
    return tf.reduce_mean(log_prob)  # 平均损失

def pace(z, prototypes, labels, num_classes=2, distance_type='euclidean'):
    """计算模态的训练速度p"""
    
    # 计算 z 到每个类原型的距离

    # 确保 labels 是整数类型
    labels = tf.cast(labels, tf.int32)
    distances = [distance(z[i], prototypes[i], distance_type) for i in range(num_classes)]  
    distances = tf.convert_to_tensor(distances)
    # 计算 softmax
    logits = -distances
    softmax = (-1) * tf.nn.softmax(logits, axis=-1)
    # print(f"softmax: {softmax}")
    
    return tf.reduce_mean(softmax)  # 平均损失

def total_loss_function(y_true, y_pred, z_0, z_1, prototypes_0, prototypes_1, ce_loss_fn=tf.keras.losses.CategoricalCrossentropy(), alpha=0.8, beta=0.1, gamma=0.1, distance_type='euclidean'):
    """组合损失函数 L_acc"""
    # 常规交叉熵损失 L_CE
    ce_loss = ce_loss_fn(y_true, y_pred)
    
    # 原型交叉熵损失 L^0_PCE 和 L^1_PCE
    pce_loss_0 = prototype_cross_entropy_loss(z_0, prototypes_0, y_true, 2, distance_type)
    pce_loss_1 = prototype_cross_entropy_loss(z_1, prototypes_1, y_true)

    p_0 = pace(z_0, prototypes_0, y_true, 2, distance_type)
    p_1 = pace(z_1, prototypes_1, y_true, 2, distance_type)
    
    # 总损失 L_acc
    total_loss = ce_loss + alpha * (beta * pce_loss_0 + gamma * pce_loss_1)
    
    return total_loss, p_0, p_1


if __name__ == "__main__":
    # 示例用法
    # 假设你有以下数据：
    # z_0, z_1: 来自两个模态的中间特征
    # prototypes_0, prototypes_1: 来自两个模态的类原型
    # y_true: 真实标签
    # y_pred: 模型预测的标签
    # ce_loss_fn: 交叉熵损失函数
    # alpha, beta, gamma: 超参数


    # 计算总损失
    # loss = total_loss_function(y_true, y_pred, z_0, z_1, prototypes_0, prototypes_1, alpha=0.5, beta=0.7, gamma=0.8, ce_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy())
    
    # 示例用法
    z = tf.random.normal((32, 128))  # 假设 z 是 (batch_size, feature_dim) 的特征
    labels = tf.one_hot(tf.random.uniform((32,), maxval=10, dtype=tf.int32), depth=10)  # 独热编码标签 (batch_size, num_classes)
    num_classes = 10

    # 计算原型
    prototypes, z = compute_prototypes(z, labels, num_classes)
    print("原型形状: ", prototypes.shape)

    pce_loss_0 = prototype_cross_entropy_loss(z, prototypes, labels, num_classes=10)