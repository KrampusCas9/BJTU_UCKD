import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
import numpy as np

def feed_forward_network(embed_dim, ff_dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(ff_dim, activation='relu'),  # (batch_size, seq_len, ff_dim)
      tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, embed_dim)
  ])

# 定义Transformer块
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = feed_forward_network(embed_dim=embed_dim, ff_dim=ff_dim)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # print(f"attn_output: {attn_output.shape}")
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # print(f"ffn_output: {ffn_output.shape}")
        # print('out1.shape',out1.shape)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.token_emb(x)

        angle_rads = self.get_angles(np.arange(maxlen)[:, np.newaxis],
                                np.arange(embed_dim)[np.newaxis, :],
                                embed_dim)

        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

        return x + pos_encoding

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def get_config(self):
        self.config = {
            "token_emb": self.token_emb
        }
        return self.config
    
class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        self.config = {
            # "pos_emb": self.pos_emb
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim
        }
        return self.config


class PositionEncoding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim

    def call(self, x):
        # maxlen = tf.shape(x)[1]
        # embed_dim = tf.shape(x)[2]
        # print('maxlen', maxlen)
        # print('embed_dim', embed_dim)
        # print(np.arange(maxlen)[:, np.newaxis], np.arange(maxlen)[:, np.newaxis].shape)
        # print(np.arange(embed_dim)[np.newaxis, :].shape)
        angle_rads = self.get_angles(np.arange(self.maxlen)[:, np.newaxis], np.arange(self.embed_dim)[np.newaxis, :], self.embed_dim)

        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        #pos_encoding = angle_rads[..., np.newaxis]
        pos_encoding = angle_rads
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        print('x',x.shape,'pe',pos_encoding.shape)
        return x + pos_encoding

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def get_config(self):
        self.config = {}
        return self.config

class transformer_feature_extractor(layers.Layer):
    def __init__(self, maxlen=64, embed_dim=64, num_heads=8, ff_dim=64):
        super().__init__()

        self.maxlen=maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.embedding_layer = PositionEncoding(maxlen=maxlen, embed_dim=embed_dim)
        self.trans_block_0 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.trans_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        self.trans_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)

    def call(self, x):
        x_0 = self.embedding_layer(x)
        x_1 = self.trans_block_0(x_0)
        x_2 = self.trans_block_1(x_1)
        x_3 = self.trans_block_2(x_2)

        # print(x_1.shape, x_2.shape, x_3.shape)

        return x_1, x_2, x_3

    def get_config(self):
        self.config = {
            # "embedding_layer": self.embedding_layer,
            # "trans_block_0": self.trans_block_0,
            # "trans_block_1": self.trans_block_1,
            # "trans_block_2": self.trans_block_2,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "maxlen": self.maxlen
        }
        return self.config

class PyramidTransformer(layers.Layer):
    def __init__(self, maxlen=64, initial_embed_dim=64, num_heads=8, ff_dim=64, num_blocks=3, reduction_factor=16,dropout_rate=0.2, **kwargs):
        super().__init__()

        self.maxlen = maxlen
        self.initial_embed_dim = initial_embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.reduction_factor = reduction_factor
        self.dropout_rate = dropout_rate

        # self.embedding_layer = PositionEncoding(maxlen=maxlen, embed_dim=initial_embed_dim)
        self.embedding_layer = PositionEmbedding(maxlen=maxlen, embed_dim=initial_embed_dim)
        self.transformer_blocks = {"0": [], "1": []}

        current_embed_dim = initial_embed_dim
        for i in range(num_blocks-1):
            dropout_rate/=(i+1)
            transformer_block_0 = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
            transformer_block_1 = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
            self.transformer_blocks["0"].append(transformer_block_0)
            self.transformer_blocks["1"].append(transformer_block_1)
            current_embed_dim -= reduction_factor  # 降维
            self.transformer_blocks["0"].append(layers.Dense(current_embed_dim))  # 添加降维层
            self.transformer_blocks["1"].append(layers.Dense(current_embed_dim))

            # self.transformer_blocks["0"].append(layers.Conv2D(kernel_size=(1,reduction_factor+1),filters=1)) # 用CNN来降维
            # self.transformer_blocks[1].append(layers.Conv2D(kernel_size=(1,reduction_factor+1),filters=1))


        transformer_block_0 = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
        transformer_block_1 = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
        self.transformer_blocks["0"].append(transformer_block_0)
        self.transformer_blocks["1"].append(transformer_block_1)
        #print('self.transformer_blocks["0"]',self.transformer_blocks["0"])

    def call(self, EEG_video):
        features_1=[]
        features_2=[]
        for i in range(2): # 两个特征
            x=EEG_video[i]
            x = self.embedding_layer(x)
            #print(f'Embedding output shape: {x.shape}',i)

            for j in range(self.num_blocks*2-1):
                if i==0:
                    I='0'
                else:
                    I='1'
                x = self.transformer_blocks[I][j](x)
                print('让我看看这个Transformer的输出形状',x.shape)
                if j%2==0: # 偶数层为Transformer 块
                    if i==1:
                        features_1.append(x)
                    else:
                        features_2.append(x)

            # 如果用Conv2D降维需要这段代码
            # for j in range(self.num_blocks*2-1):
            #     print('让我看看这个Transformer的输出形状',x.shape)
            #     if j%2==0: # 偶数层为Transformer 块
            #         x = self.transformer_blocks[i][j](x)
            #         if i==1:
            #             features_1.append(x)
            #         else:
            #             features_2.append(x)
            #     else: # 如果用Conv2D降维需要这段代码
            #         x=tf.expand_dims(x, axis=-1)
            #         x = self.transformer_blocks[i][j](x)
            #         x=tf.squeeze(x, axis=-1)

        print('features_1[0].shape',features_1[0].shape)
        attention_outputs, attention_scoreses=[],[]
        j=0
        for i in range(self.num_blocks): # 生成num_blocks张特征矩阵
            query=self.transformer_blocks["0"][j].att._query_dense(features_1[i])
            #print('query.shape',query.shape)
            key=self.transformer_blocks["1"][j].att._key_dense(features_2[i])
            value=self.transformer_blocks["1"][j].att._value_dense(features_2[i])

            attention_output, attention_scores = self.transformer_blocks["0"][j].att._compute_attention(query, key, value)
            # attention_output=tf.reduce_mean(attention_output, axis=2)
            print("attention_output",attention_output.shape)
            attention_outputs.append(attention_output)
            attention_scoreses.append(attention_scores)
            j+=2
        print('PyramidTransformer end')
        print("len(attention_outputs)",len(attention_outputs),len(attention_scoreses))
        print("attention_outputs[1].shape",attention_outputs[1].shape)
        return attention_outputs,attention_scoreses

    def get_config(self):
        config = super(PyramidTransformer, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'initial_embed_dim': self.initial_embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_blocks': self.num_blocks,
            'reduction_factor': self.reduction_factor,
        })
        return config

class Single_PyramidTransformer(layers.Layer):
    '''
        单模态的金字塔Transformer
    '''
    def __init__(self, maxlen=64, initial_embed_dim=64, num_heads=8, ff_dim=64, num_blocks=3, reduction_factor=16,dropout_rate=0.2, **kwargs):
        super().__init__()

        self.maxlen = maxlen
        self.initial_embed_dim = initial_embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.reduction_factor = reduction_factor
        self.dropout_rate = dropout_rate

        self.embedding_layer = PositionEmbedding(maxlen=maxlen, embed_dim=initial_embed_dim)
        # self.embedding_layer = PositionEncoding(maxlen, initial_embed_dim)
        self.transformer_blocks = []

        current_embed_dim = initial_embed_dim
        for i in range(num_blocks-1):
            dropout_rate/=(i+1)
            transformer_block = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
            self.transformer_blocks.append(transformer_block)
            current_embed_dim -= reduction_factor  # 降维
            self.transformer_blocks.append(layers.Dense(current_embed_dim))  # 添加降维层

        transformer_block = TransformerBlock(embed_dim=current_embed_dim, num_heads=num_heads, ff_dim=ff_dim,dropout_rate=dropout_rate)
        self.transformer_blocks.append(transformer_block)
        # print('self.transformer_blocks',self.transformer_blocks)

    def call(self, x):
        features=[]
        # print('x.shape', x.shape)
        x = self.embedding_layer(x)
        # print('after embedding', x.shape)
        # print('in trans block', x.shape)
        for j in range(self.num_blocks*2-1):
            x = self.transformer_blocks[j](x)
            #print('让我看看这个Transformer的输出形状',x.shape)
            if j%2==0: # 偶数层为Transformer 块
                features.append(x)

        return features

    def get_config(self):
        config = super(Single_PyramidTransformer, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'initial_embed_dim': self.initial_embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_blocks': self.num_blocks,
            'reduction_factor': self.reduction_factor,
            "dropout_rate": self.dropout_rate,
        })
        return config








if __name__ == "__main__":
    # 示例输入数据
    batch_size = 15
    sequence_length = 5
    feature_dimension = 64
    embed_dim = 64
    num_heads = 8

    # 随机生成输入特征
    feature1 = np.random.rand(batch_size, sequence_length, feature_dimension).astype(np.float32)
    feature2 = np.random.rand(batch_size, sequence_length, feature_dimension).astype(np.float32)

    pyramid_transformer = PyramidTransformer(maxlen=64, initial_embed_dim=64, num_heads=8, ff_dim=64, num_blocks=3, reduction_factor=16)

    attention_outputs, attention_scoreses = pyramid_transformer([feature1, feature2])


    print("Attention Output Shape:", len(attention_outputs))
    for i in attention_outputs:
        print(i.shape)
    print("Attention Scores Shape:", attention_scoreses.shape)