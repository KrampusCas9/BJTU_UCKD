import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, LeakyReLU


class Classifier(layers.Layer):
    def __init__(self, hide_dim, output_dim):
        super().__init__()

        self.hide_dim = hide_dim
        self.output_dim = output_dim

        self.dense_0 = layers.Dense(hide_dim)
        self.dense_1 = layers.Dense(hide_dim)
        self.dense_2 = layers.Dense(hide_dim)
        self.final_dense = layers.Dense(output_dim)
        self.dropout_1 = layers.Dropout(0.5)
        #self.dropout_2 = layers.Dropout(0.5)

    def call(self, inputs):
        x = self.dropout_1(inputs)
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        #x = self.dropout_2(x)
        x = self.final_dense(x)
        return x

    def get_config(self):
        self.config = {
            # "dense_0": self.dense_0,
            # "dense_1": self.dense_1,
            # "dense_2": self.dense_2,
            # "final_dense": self.final_dense,
            "hide_dim": self.hide_dim,
            "output_dim": self.output_dim
        }
        return self.config

class EEG_Head(layers.Layer):
    def __init__(self):
        super().__init__()
        self.padding = "same"
        self.activation = tf.nn.relu
        self.temporal_cnn_layer = layers.Conv2D(kernel_size=(1, 32), filters=64, strides=(1, 8),
                                                kernel_regularizer=keras.regularizers.l2(0.001))
        self.activate_layer = layers.Activation(activation=self.activation)
        self.bm = layers.BatchNormalization()
        self.maxpooling = layers.MaxPool2D(pool_size=(1, 8), strides=8)
        self.spatoal_fusion_layer = layers.Conv2D(kernel_size=(28, 1), filters=64, strides=(14, 1),
                                                  kernel_regularizer=keras.regularizers.l2(0.001))
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        x = self.temporal_cnn_layer(inputs)
        x = self.bm(x)
        x = self.activate_layer(x)
        #print(f"CNN_1_output: {x.shape}")
        x = self.spatoal_fusion_layer(x)
        #print('x.shape',x.shape)
        x = tf.squeeze(x, axis=1)
        x = self.dropout(x)
        #print(f"CNN_2_output: {x.shape}")
        return x

    def get_config(self):
        self.config = {
            # "temporal_cnn_layer": self.temporal_cnn_layer,
            # "activate_layer": self.activate_layer,
            # "bm": self.bm,
            # "maxpooling": self.maxpooling,
            # "spatoal_fusion_layer": self.spatoal_fusion_layer,
            # "reshape": self.reshape
        }
        return self.config
    
class video_Head(layers.Layer):
    def __init__(self):
        super().__init__()
        self.padding = "same"
        self.activation = tf.nn.relu
        self.temporal_cnn_layer = layers.Conv2D(kernel_size=(1, 32), filters=64, strides=(1, 8),
                                                kernel_regularizer=keras.regularizers.l2(0.001))
        self.activate_layer = layers.Activation(activation=self.activation)
        self.bm = layers.BatchNormalization()
        self.spatoal_fusion_layer = layers.Conv2D(kernel_size=(5, 33), filters=64, strides=(1, 1),
                                                  kernel_regularizer=keras.regularizers.l2(0.001))
        self.conv1x1 = layers.Conv2D(kernel_size=(1, 1), filters=1)
        #self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        print('init inputs',inputs.shape)
        x = self.temporal_cnn_layer(inputs)
        print(f"CNN_1_output: {x.shape}")
        x = self.bm(x)
        x = self.activate_layer(x)
        x = self.spatoal_fusion_layer(x)
        print(f"CNN_2_output: {x.shape}")
        x = tf.transpose(x, perm = (0,2,3,1))
        print('after transpose', x.shape)
        x = self.conv1x1(x)
        print('after conv1x10', x.shape)
        x = tf.squeeze(x, axis=-1)
        print('after squeeze', x.shape)
        x = self.dropout(x)
        return x

    def get_config(self):
        self.config = {
            # "temporal_cnn_layer": self.temporal_cnn_layer,
            # "activate_layer": self.activate_layer,
            # "bm": self.bm,
            # "spatoal_fusion_layer": self.spatoal_fusion_layer
        }
        return self.config

class SamplingLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(SamplingLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = layers.Dense(self.output_dim, activation='relu', kernel_initializer='he_uniform')
        self.dropout = layers.Dropout(0.5)
    def call(self, inputs):
        # 调整输入数据的最后一个维度
        return self.dropout(self.dense(inputs))

    def get_config(self):
        config = super(SamplingLayer, self).get_config()
        config.update({
            'output_dim': self.output_dim
        })
        return config

class MLP(layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            'hidden_dim': self.fc1.units,
            'output_dim': self.fc2.units
        })
        return config
    
class ConvBlock(layers.Layer):
    def __init__(self, out_chan, kernel, step, pool):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(out_chan, kernel, strides=step, padding='same', use_bias=False)
        self.activation = LeakyReLU()
        self.pool = AveragePooling2D(pool_size=(1, pool), strides=(1, pool))

    def call(self, inputs):
        #print('in block inputs',inputs.shape)
        x = self.conv(inputs)
        #print('in block after conv',x.shape)
        x = self.activation(x)
        x = self.pool(x)
        #print('in block after pool', x.shape)
        return x

class DynamicAveragePooling(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(DynamicAveragePooling, self).__init__(**kwargs)
        self.axis = axis  # 指定你要池化的维度

    def call(self, inputs):
        # 动态获取输入在指定维度的大小
        input_shape = tf.shape(inputs)
        pool_size = input_shape[self.axis]
        
        # 仅在指定的维度上进行平均池化
        if self.axis == 1:  # 如果要对height维度池化
            return tf.reduce_mean(inputs, axis=1, keepdims=True)
        elif self.axis == 2:  # 如果要对width维度池化
            return tf.reduce_mean(inputs, axis=2, keepdims=True)
        else:
            raise ValueError("This pooling layer currently only supports axis 1 (height) or axis 2 (width).")