import tensorflow as tf
from tensorflow.keras import layers, Model
from modules.BasicModule import Classifier, EEG_Head, video_Head, SamplingLayer
from modules.Transformer_model import transformer_feature_extractor, PyramidTransformer, Single_PyramidTransformer
from modules.Attention import MultiDepthAttentionFusion
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU
from modules.BasicModule import Classifier, EEG_Head, video_Head, SamplingLayer, ConvBlock, DynamicAveragePooling


class VideoModel(Model):
    def __init__(self):
        super(VideoModel, self).__init__()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        # 定义输入
        self.input_ori_video = layers.Input((5, 768, 1), name="video_input")

        # 定义模型层
        self.video_head = video_Head()
        #self.transformer_feature_extractor = transformer_feature_extractor()
        #self.multi_depth_attention_fusion = MultiDepthAttentionFusion(64, 64)
        self.pyramid_transformer = Single_PyramidTransformer()
        self.sampling_layer1 = SamplingLayer(output_dim=48)
        self.sampling_layer2 = SamplingLayer(output_dim=48)
        self.sampling_layer3 = SamplingLayer(output_dim=48)
        self.conv2d = layers.Conv2D(filters=1, kernel_size=(14,1))
        self.flatten = layers.Flatten(name="att_feature")
        

        # 使用 AveragePooling1D 进行降维
        # pool_size 设置为 2，这意味着每两个元素取平均，从而将 2304 维降到 1152 维
        # strides 设置为 1 表示每次移动一个元素（也可以尝试其他值如 2 来更快降维）
        self.avg_pool_layer = layers.AveragePooling1D(pool_size=769, strides=1)

        self.classifier = Classifier(hide_dim=128, output_dim=2)
        self.softmax = layers.Activation(activation="softmax", name="cls")


    def call(self, inputs):
        input_ori_video = inputs

        # 前向传播逻辑
        video_feature = self.video_head(input_ori_video)
        attention_outputs = self.pyramid_transformer(video_feature)
        video_feature_1, video_feature_2, video_feature_3 = attention_outputs
        # # 使用 tf.unstack 将张量解包
        # video_feature_1, video_feature_2, video_feature_3 = tf.unstack(attention_outputs, num=3)

        video_feature_1 = self.sampling_layer1(video_feature_1)
        video_feature_2 = self.sampling_layer2(video_feature_2)
        video_feature_3 = self.sampling_layer3(video_feature_3)

        att_feature = video_feature_1 + video_feature_2 + video_feature_3
        
        att_feature = tf.expand_dims(att_feature, axis=-1)
        att_feature = self.conv2d(att_feature)
        print('video_feature',video_feature_1.shape)
        print('att_feature.shape',att_feature.shape)
        att_feature = self.flatten(att_feature)
        
        # 将输入 reshaped 为 (batch_size, 2304, 1)，适合应用 1D 池化
        att_feature = tf.expand_dims(att_feature, axis=-1)
        att_feature = self.avg_pool_layer(att_feature)
        print('after avgpool',att_feature.shape)
        att_feature = tf.squeeze(att_feature, axis=-1)
        logits = self.classifier(att_feature)
        cls = self.softmax(logits)

        return cls, logits, att_feature # 返回变量大于1就不能用fit函数训练了，需要自定义train_step

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            pred, _, _ = self(inputs, training=True)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels,pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)
    
        # 返回损失值和度量指标
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }
    

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.acc_metric]

    def test_step(self, data):
        inputs, labels = data
        pred, _, _ = self(inputs, training=False)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(labels,pred)

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        # 返回损失值和度量指标
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }



class EEGModel(Model):
    def __init__(self):
        super(EEGModel, self).__init__()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        # 定义输入
        self.input_ori_eeg = layers.Input((28, 512, 1), name="eeg_input")

        # 定义模型层
        self.eeg_head = EEG_Head()
        #self.transformer_feature_extractor = transformer_feature_extractor()
        #self.multi_depth_attention_fusion = MultiDepthAttentionFusion(64, 64)
        self.pyramid_transformer = Single_PyramidTransformer()
        self.sampling_layer1 = SamplingLayer(output_dim=48)
        self.sampling_layer2 = SamplingLayer(output_dim=48)
        self.sampling_layer3 = SamplingLayer(output_dim=48)
        self.flatten = layers.Flatten(name="att_feature")
        self.conv2d = layers.Conv2D(filters=1, kernel_size=(14,1))
        self.classifier = Classifier(hide_dim=128, output_dim=2)
        self.softmax = layers.Activation(activation="softmax", name="cls")


    def call(self, inputs):
        input_ori_eeg = inputs

        # 前向传播逻辑
        eeg_feature = self.eeg_head(input_ori_eeg)
        attention_outputs = self.pyramid_transformer(eeg_feature)
        eeg_feature_1, eeg_feature_2, eeg_feature_3 = attention_outputs
        # # 使用 tf.unstack 将张量解包
        # eeg_feature_1, eeg_feature_2, eeg_feature_3 = tf.unstack(attention_outputs, num=3)

        eeg_feature_1 = self.sampling_layer1(eeg_feature_1)
        eeg_feature_2 = self.sampling_layer2(eeg_feature_2)
        eeg_feature_3 = self.sampling_layer3(eeg_feature_3)

        att_feature = eeg_feature_1 + eeg_feature_2 + eeg_feature_3
        
        att_feature = tf.expand_dims(att_feature, axis=-1)
        att_feature = self.conv2d(att_feature)
        #print('eeg_feature',eeg_feature_1.shape)
        att_feature = self.flatten(att_feature)
        logits = self.classifier(att_feature)
        cls = self.softmax(logits)

        return cls, logits # 返回变量大于1就不能用fit函数训练了，需要自定义train_step


    #@tf.function
    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            pred, _ = self(inputs, training=True)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels,pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        # 返回损失值和度量指标
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.acc_metric]

    #@tf.function
    def test_step(self, data):
        inputs, labels = data
        pred, _ = self(inputs, training=False)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(labels,pred)

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        # 返回损失值和度量指标
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }

class Tsception(Model):
    def __init__(self, num_classes, Chans, Samples, sampling_rate, num_T, num_S, hidden, dropout_rate, pool=8):
        super(Tsception, self).__init__()

        '''
        input_size: 输入数据的维度,(chans, samples, 1)

        num_classes   : 输出类别的数量 (二分类问题中为 2)
        
        Samples       : 每个通道的采样点数 (输入数据的时间维度)

        sampling_rate : EEG 采样率 (每秒采样点数，用于计算卷积核大小)

        num_T         : 时域卷积核的数量 (提取时间动态性特征)

        num_S         : 空域卷积核的数量 (提取空间不对称性特征)

        hidden        : 全连接层的隐藏单元数 (控制模型复杂度)
        
        dropout_rate  : Dropout 丢弃率 (防止过拟合)
        '''
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        self.inception_window = [0.5, 0.25, 0.125]
        self.num_T = num_T
        self.num_S = num_S
        self.Chans = Chans
        self.Samples = Samples
        self.sampling_rate = sampling_rate
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.pool = pool

        # 定义时域卷积块
        self.conv1 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[0])), 1, pool)
        self.conv2 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[1])), 1, pool)
        self.conv3 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[2])), 1, pool)

        # 定义空域卷积块
        self.sconv1 = ConvBlock(num_S, (Chans, 1), (Chans, 1), int(pool*0.25))
        self.sconv2 = ConvBlock(num_S, (int(Chans*0.5), 1), (int(Chans*0.5), 1), int(pool*0.25))

        # 定义fusion layer
        self.fusion_conv = ConvBlock(num_S, (3, 1), (3, 1), 4)

        # 定义 BatchNormalization 层
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.batchnorm3 = BatchNormalization()

        # 定义全局平均池化层(将axis=2维度变成1)
        self.global_avg_pool = DynamicAveragePooling(axis=2) 

        # 定义全连接层
        self.flatten = Flatten()
        self.fc1 = Dense(hidden, activation='relu', use_bias=False)
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Dense(num_classes, activation='softmax', use_bias=False)

    def call(self, inputs, training=False):
        # 时域卷积
        x1 = self.conv1(inputs)
        print('x1',x1.shape)
        x2 = self.conv2(inputs)
        print('x2',x2.shape)
        x3 = self.conv3(inputs)
        print('x3',x3.shape)
        
        # 在height维度上进行拼接
        x = concatenate([x1, x2, x3], axis=2)
        x = self.batchnorm1(x)
        print('x',x.shape)

        # 空域卷积
        y1 = self.sconv1(x)
        y2 = self.sconv2(x)
        print('y1',y1.shape)
        print('y2',y2.shape)

        # 在width维度上进行拼接
        y = concatenate([y1, y2], axis=1)
        y = self.batchnorm2(y)
        print('y',y.shape)

        # Fusion layer
        z = self.fusion_conv(y)
        print('z',z.shape)
        z = self.batchnorm3(z)

        # 全局平均池化层和全连接层
        z = self.global_avg_pool(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = self.dropout(z, training=training)
        z = self.fc2(z)
        return z
    
    #@tf.function
    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            pred = self(inputs, training=True)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_fn(labels,pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        # 返回损失值和度量指标
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.acc_metric]
    
    #@tf.function
    def test_step(self, data):
        inputs, labels = data
        pred = self(inputs, training=False)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(labels,pred)

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        # 返回损失值和度量指标
        #return loss, {m.name: m.result() for m in self.metrics}
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_metric.result(),
        }
    
if __name__ == "__main__":
    # 使用示例
    model = Tsception(num_classes=2, Chans=28, Samples=512, sampling_rate=128, num_T=64, num_S=32, hidden=32, dropout_rate=0.5)
    model.compile()
    model.build(input_shape=(10, 28, 512, 1))
    model.summary()

    # 创建输入数据
    inputs = tf.random.normal([10, 28, 512, 1])

    # 方式 1: 直接调用模型对象
    outputs = model(inputs)
    #print(outputs)
