# creates a model to reconstruct 3d depth from 2d data

import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="custom_layer")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__() # initialize parent class
        self.pos_encoding = self.positional_encoding(seq_len, d_model)
    
    def get_config(self):
        config = super.get_config()
        config.update({"seq_len": self.pos_encoding.shape[0], "d_model": self.pos_encoding.shape[1]}) # update configs
        return config

    def positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(np.arange(seq_len)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]  # shape (1, seq_len, d_model)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        return pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    def call(self, x):
        pos = tf.cast(self.pos_encoding[:, :tf.shape(x)[1], :], x.dtype)
        return x + pos

@tf.keras.utils.register_keras_serializable(package="custom_model")
class Reconstructor(tf.keras.Model):

    """a transformer to reconstruct depth from 2d data"""

    def __init__(self, seq_len=120, d_model=64, **kwargs):
        super().__init__(**kwargs)

        # positional encoding
        self.pos_encoding = PositionalEncoding(seq_len, d_model)

        # dense to map input features -> d_model
        self.input_dense = tf.keras.layers.Dense(d_model) 

        # first lstm block
        self.lstm1 = tf.keras.layers.LSTM(64, activation="tanh", return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # first transformer block
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # second lstm block
        self.lstm3 = tf.keras.layers.LSTM(256, activation="tanh", return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(512, activation="tanh", return_sequences=True)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # second transformer block
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # third lstm block
        self.lstm5 = tf.keras.layers.LSTM(256, activation="tanh", return_sequences=True)
        self.lstm6 = tf.keras.layers.LSTM(512, activation="tanh", return_sequences=True)
        self.bn3 = tf.keras.layers.BatchNormalization()

        # third transformer block
        self.mha3 = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=d_model)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # feed-forward network
        self.ffn_dense1 = tf.keras.layers.Dense(128, activation="gelu")
        self.ffn_dense2 = tf.keras.layers.Dense(256, activation="gelu")
        self.ffn_dense3 = tf.keras.layers.Dense(512, activation="gelu")
        self.ffn_dense4 = tf.keras.layers.Dense(1)  # output Z for each frame

    def call(self, x, training=False):

        # project input 2D -> d_model
        x = self.input_dense(x)

        # add positional encoding
        x = self.pos_encoding(x)

        # temporal block 1
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.bn1(x)

        # transformer block 1
        attn_output1 = self.mha1(query=x, value=x, key=x)
        attn_output1 = self.dropout1(attn_output1, training=training)
        x = self.ln1(x + attn_output1)  # residual connection + normalization

        # temporal block 2
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.bn2(x)

        # transformer block 2
        attn_output2 = self.mha2(query=x, value=x, key=x)
        attn_output2 = self.dropout2(attn_output2, training=training)
        x = self.ln2(x + attn_output2)  # residual connection + normalization

        # temporal block 3
        x = self.lstm5(x)
        x = self.lstm6(x)
        x = self.bn3(x)

        # transformer block 3
        attn_output3 = self.mha3(query=x, value=x, key=x)
        attn_output3 = self.dropout3(attn_output3, training=training)
        x = self.ln3(x + attn_output3)

        # feed-forward
        x = self.ffn_dense1(x)
        x = self.ffn_dense2(x)
        x = self.ffn_dense3(x)
        x = self.ffn_dense4(x)

        return x
    
    def build(self, input_shape):
        # this will initialize all layers with the input shape
        super().build(input_shape)