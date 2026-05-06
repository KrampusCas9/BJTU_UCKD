import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import optimizers
import yaml
import wandb
from modules.BasicModule import Classifier, ConvBlock, EEG_Head, video_Head, SamplingLayer
from modules.Transformer_model import PyramidTransformer, Single_PyramidTransformer
from model import VideoModel_front
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU

class Tsception_TA_part_1(Model):
    def __init__(self, Chans, Samples, sampling_rate, num_T, num_S, hidden, dropout_rate, pool=8):
        super(Tsception_TA_part_1, self).__init__()
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        self.inception_window = [0.5, 0.125] 
        self.num_T = num_T
        self.num_S = num_S
        self.Chans = Chans
        self.Samples = Samples
        self.sampling_rate = sampling_rate
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.pool = pool

        # 定义时域卷积块 # 三层变两层
        self.conv1 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[0])), 1, pool)
        self.conv2 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[1])), 1, pool)

        # 使用 ZeroPadding2D，在宽度方向上填充
        self.padded_tensor = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (32, 32)))

        # 定义空域卷积块 # 两层变一层
        self.sconv = ConvBlock(num_S, (10, 1), (10, 1), int(pool*0.25))

        # 定义fusion layer
        self.fusion_conv = ConvBlock(num_S, (3, 1), (3, 1), 4)

        # 定义 BatchNormalization 层
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.batchnorm3 = BatchNormalization()

        # 定义全局平均池化层(将axis=2维度变成1)
        #self.global_avg_pool = DynamicAveragePooling(axis=2) 

        # 定义全连接层
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        # 时域卷积
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        
        # 在height维度上进行拼接
        x = concatenate([x1, x2], axis=2)
        x = self.batchnorm1(x)
        #print('before padding',x.shape)
        x = self.padded_tensor(x)
        #print('after padding，空域卷积前',x.shape)
        # 空域卷积
        y = self.sconv(x)
        #print('空域卷积后',y.shape)
        y = self.batchnorm2(y)

        # Fusion layer
        z = self.fusion_conv(y)
        z = self.batchnorm3(z)

        z = self.flatten(z)
        return z
    
class Tsception_TA_part_2(Model):
    def __init__(self, Chans, Samples, sampling_rate, num_T, num_S, hidden, dropout_rate, pool=8):
        super(Tsception_TA_part_2, self).__init__()
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        self.inception_window = [0.5] 
        self.num_T = num_T
        self.num_S = num_S
        self.Chans = Chans
        self.Samples = Samples
        self.sampling_rate = sampling_rate
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.pool = pool

        # 定义时域卷积块 # 三层变两层 # 两层变一层
        self.conv1 = ConvBlock(num_T, (1, int(sampling_rate * self.inception_window[0])), 1, pool)

        # 使用 ZeroPadding2D，在宽度方向上填充
        self.padded_tensor = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (32, 32)))

        # 定义空域卷积块 # 两层变一层
        self.sconv = ConvBlock(num_S, (10, 1), (10, 1), int(pool*0.25))

        # 定义fusion layer
        self.fusion_conv = ConvBlock(num_S, (3, 1), (3, 1), 4)

        # 定义 BatchNormalization 层
        self.batchnorm1 = BatchNormalization()
        self.batchnorm2 = BatchNormalization()
        self.batchnorm3 = BatchNormalization()

        # 定义全局平均池化层(将axis=2维度变成1)
        #self.global_avg_pool = DynamicAveragePooling(axis=2) 

        # 定义全连接层
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        # 时域卷积
        x = self.conv1(inputs)
        
        x = self.batchnorm1(x)
        #print('before padding',x.shape)
        x = self.padded_tensor(x)
        #print('after padding，空域卷积前',x.shape)
        # 空域卷积
        y = self.sconv(x)
        #print('空域卷积后',y.shape)
        y = self.batchnorm2(y)

        # Fusion layer
        z = self.fusion_conv(y)
        z = self.batchnorm3(z)

        z = self.flatten(z)
        return z


class TA_MultiModalModel_1(Model):
    def __init__(self):
        super(TA_MultiModalModel_1, self).__init__()
        

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        self.video_model = VideoModel_front()
        self.eeg_model = Tsception_TA_part_1(Chans=28, Samples=512, sampling_rate=128, num_T=64, num_S=32, hidden=32, dropout_rate=0.5)

        self.classifier = Classifier(hide_dim=128, output_dim=2)
        

    def call(self, inputs):
        input_ori_video, input_ori_eeg = inputs

        video_feature = self.video_model(input_ori_video)

        eeg_feature = self.eeg_model(input_ori_eeg)

        # print("video_feature.shape",video_feature.shape)
        # print("eeg_feature.shape",eeg_feature.shape)

        # crossfusion_feature = video_feature + eeg_feature
        crossfusion_feature = tf.concat([video_feature, eeg_feature], axis=1)

        flatten_feature = layers.Flatten()(crossfusion_feature)
        
        logit_feature = self.classifier(flatten_feature)
        cls = layers.Activation(activation="softmax")(logit_feature)
        # print('TA model', flatten_feature.shape)
        #print('cls',cls.shape)
        return cls, logit_feature
    
    
    #@tf.function
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
    
    #@tf.function
    def test_step(self, data):
        inputs, labels = data
        pred, _, _ = self(inputs, training=False)
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
    

class TA_MultiModalModel_2(Model):
    def __init__(self):
        super(TA_MultiModalModel_2, self).__init__()
        

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        self.video_model = VideoModel_front()
        self.eeg_model = Tsception_TA_part_2(Chans=28, Samples=512, sampling_rate=128, num_T=64, num_S=32, hidden=32, dropout_rate=0.5)

        self.classifier = Classifier(hide_dim=128, output_dim=2)
        

    def call(self, inputs):
        input_ori_video, input_ori_eeg = inputs

        video_feature = self.video_model(input_ori_video)

        eeg_feature = self.eeg_model(input_ori_eeg)

        # print("video_feature.shape",video_feature.shape)
        # print("eeg_feature.shape",eeg_feature.shape)

        # crossfusion_feature = video_feature + eeg_feature
        crossfusion_feature = tf.concat([video_feature, eeg_feature], axis=1)

        flatten_feature = layers.Flatten()(crossfusion_feature)
        
        logit_feature = self.classifier(flatten_feature)
        cls = layers.Activation(activation="softmax")(logit_feature)
        # print('TA model', flatten_feature.shape)
        #print('cls',cls.shape)
        return cls, logit_feature
    
    
    #@tf.function
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
    
    #@tf.function
    def test_step(self, data):
        inputs, labels = data
        pred, _, _ = self(inputs, training=False)
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

    model = TA_MultiModalModel_1()

    # 编译模型
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 打印模型结构
    model.build(input_shape=[(10, 5, 768, 1),(10, 28, 512, 1)])
    model.summary()