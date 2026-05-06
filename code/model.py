import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU
import yaml
import wandb
from modules.BasicModule import Classifier, EEG_Head, video_Head, SamplingLayer, ConvBlock, DynamicAveragePooling
from modules.Transformer_model import PyramidTransformer, Single_PyramidTransformer
from PMR_loss import compute_prototypes, total_loss_function
import contrastive_losses


class VideoModel_front(Model):
    def __init__(self):
        super(VideoModel_front, self).__init__()

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric=tf.keras.metrics.CategoricalAccuracy()

        # 定义输入
        self.input_ori_video = layers.Input((5, 768, 1), name="video_input")

        # 定义模型层
        self.video_head = video_Head()
        #self.transformer_feature_extractor = transformer_feature_extractor()
        #self.multi_depth_attention_fusion = MultiDepthAttentionFusion(64, 64)
        self.pyramid_transformer = Single_PyramidTransformer(maxlen=61)
        self.sampling_layer1 = SamplingLayer(output_dim=48)
        self.sampling_layer2 = SamplingLayer(output_dim=48)
        self.sampling_layer3 = SamplingLayer(output_dim=48)
        self.conv2d = layers.Conv2D(filters=1, kernel_size=(14,1))
        self.flatten = layers.Flatten(name="att_feature")
        self.fc = layers.Dense(768)
        #self.classifier = Classifier(hide_dim=128, output_dim=2)

    def call(self, inputs):
        #input_ori_video, input_hard_label, input_soft_label, input_soft_feature = inputs
        input_ori_video = inputs

        # 前向传播逻辑
        # print('before video head', input_ori_video.shape)
        video_feature = self.video_head(input_ori_video)
        # print('after video head', video_feature.shape)
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
        #print('video_feature',video_feature_1.shape)
        att_feature = self.flatten(att_feature)
        att_feature = self.fc(att_feature)
        return att_feature # VideoModel的中间变量

class Tsception_front(Model):
    def __init__(self, num_classes, Chans, Samples, sampling_rate, num_T, num_S, hidden, dropout_rate, pool=8):
        super(Tsception_front, self).__init__()
        
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
        #self.global_avg_pool = DynamicAveragePooling(axis=2) 

        # 定义全连接层
        self.flatten = Flatten()
        #self.fc1 = Dense(hidden, activation='relu', use_bias=False)
        #self.dropout = Dropout(dropout_rate)
        #self.fc2 = Dense(num_classes, activation='softmax', use_bias=False)

    def call(self, inputs, training=False):
        # 时域卷积
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        
        # 在height维度上进行拼接
        x = concatenate([x1, x2, x3], axis=2)
        x = self.batchnorm1(x)

        # 空域卷积
        y1 = self.sconv1(x)
        y2 = self.sconv2(x)

        # 在width维度上进行拼接
        y = concatenate([y1, y2], axis=1)
        y = self.batchnorm2(y)

        # Fusion layer
        z = self.fusion_conv(y)
        z = self.batchnorm3(z)

        # 全局平均池化层和全连接层
        #print('before avg_pool',z.shape)
        # z = self.global_avg_pool(z)
        # print("after avg_pool",z.shape)
        z = self.flatten(z)
        #print('flatten',z.shape)
        # z = self.fc1(z)
        # print('after f1',z.shape)
        # z = self.dropout(z, training=training)
        # print("after dropout",z.shape)
        #z = self.fc2(z)
        return z
    
class TS_MultiModalModel(Model):
    def __init__(self, args):
        super(TS_MultiModalModel, self).__init__()

        self.args = args

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy()

        if self.args.PMR_loss:
            self.p_0_sum = tf.keras.metrics.Sum()
            self.p_1_sum = tf.keras.metrics.Sum()
            self.beta = 0
            self.gamma = 0
        
        self.video_model = VideoModel_front()
        self.eeg_model = Tsception_front(num_classes=2, Chans=28, Samples=512, sampling_rate=128, num_T=64, num_S=32, hidden=32, dropout_rate=0.5)

        self.classifier = Classifier(hide_dim=128, output_dim=2)


    def call(self, inputs):
        input_ori_video, input_ori_eeg = inputs

        # input_ori_video = inputs['video']
        # input_ori_eeg = inputs['eeg']
        # input_ori_eeg = tf.expand_dims(input_ori_eeg, axis=-1)
        # input_ori_video = tf.expand_dims(input_ori_video, axis=-1)
        # print("input_ori_video.shape", input_ori_video.shape)
        # print("input_ori_eeg.shape", input_ori_eeg.shape)

        video_feature = self.video_model(input_ori_video)

        eeg_feature = self.eeg_model(input_ori_eeg)

        # print("video_feature.shape",video_feature.shape)
        # print("eeg_feature.shape",eeg_feature.shape)

        # crossfusion_feature = video_feature + eeg_feature
        crossfusion_feature = tf.concat([video_feature, eeg_feature], axis=1)

        flatten_feature = layers.Flatten()(crossfusion_feature)
        
        logit_feature = self.classifier(flatten_feature)
        cls = layers.Activation(activation="softmax")(logit_feature)
        #print('cls',cls.shape)
        # print('teacher model', flatten_feature.shape)
        return cls, video_feature, eeg_feature, logit_feature, flatten_feature
    
    
    #@tf.function
    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            pred, video_feature, eeg_feature, _, _ = self(inputs, training=True)
            if self.args.PMR_loss:
            # 计算当前批次的类原型
                prototypes_0, video_classes = compute_prototypes(video_feature, labels)
                prototypes_1, eeg_classes = compute_prototypes(eeg_feature, labels)
                loss, p_0, p_1 = total_loss_function(labels, pred, video_classes, eeg_classes
                                                     , prototypes_0, prototypes_1
                                                     , alpha=self.args.alpha, beta=self.beta, gamma=self.gamma)

            elif self.args.contrastive_learning:
                video_contrastive_loss = contrastive_losses.max_margin_contrastive_loss(video_feature, labels, metric='cosine')
                eeg_contrastive_loss = contrastive_losses.max_margin_contrastive_loss(eeg_feature, labels, metric='cosine')
                cls_loss_fn = tf.keras.losses.CategoricalCrossentropy()
                cls_loss = cls_loss_fn(labels,pred)
                loss =  cls_loss + video_contrastive_loss + eeg_contrastive_loss
            else:
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
                loss = loss_fn(labels,pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.acc_metric.update_state(labels, pred)
        # 更新损失追踪器
        self.loss_tracker.update_state(loss)

        if self.args.PMR_loss:
            self.p_0_sum.update_state(p_0)
            self.p_1_sum.update_state(p_1)

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
        pred, _, _, _, _ = self(inputs, training=False)
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
    

class PMRCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(PMRCallback, self).__init__()
        self.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        """在每个 epoch 结束时，调整损失加权系数"""

        rho = self.model.p_0_sum.result().numpy() / self.model.p_1_sum.result().numpy()

        if rho <1:
            self.model.beta = tf.clip_by_value(1/rho - 1, 0, 1)
            self.model.gamma = 0
        else :
            self.model.beta = 0
            self.model.gamma = tf.clip_by_value(rho - 1, 0, 1)

        self.model.p_0_sum.reset_state()
        self.model.p_1_sum.reset_state()