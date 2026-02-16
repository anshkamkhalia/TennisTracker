# creates a model to reconstruct 3d depth from 2d data

import tensorflow as tf
import numpy as np

@tf.keras.saving.register_keras_serializable(package="custom_layer")
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
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

@tf.keras.saving.register_keras_serializable(package="custom_model")
class Reconstructor(tf.keras.Model):  # note: tf.keras.Model, not tf.keras.model

    """a transformer to reconstruct depth from 2d data"""

    def __init__(self, seq_len=60, d_model=64, **kwargs):
        super().__init__(**kwargs)

        # positional encoding
        self.pos_encoding = PositionalEncoding(seq_len, d_model)

        # first transformer block
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # second transformer block
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # feed-forward network
        self.ffn_dense1 = tf.keras.layers.Dense(128, activation="gelu")
        self.ffn_dense2 = tf.keras.layers.Dense(256, activation="gelu")
        self.ffn_dense3 = tf.keras.layers.Dense(1)  # output Z for each frame

    def call(self, x, training=False):

        # project input 2D -> d_model
        x = tf.keras.layers.Dense(self.pos_encoding.pos_encoding.shape[-1])(x)  # shape -> (batch, seq_len, d_model)

        # add positional encoding
        x = self.pos_encoding(x)

        # transformer block 1
        attn_output1 = self.mha1(query=x, value=x, key=x)
        attn_output1 = self.dropout1(attn_output1, training=training)
        x = self.ln1(x + attn_output1)  # residual connection + normalization

        # transformer block 2
        attn_output2 = self.mha2(query=x, value=x, key=x)
        attn_output2 = self.dropout2(attn_output2, training=training)
        x = self.ln2(x + attn_output2)  # residual connection + normalization

        # feed-forward
        x = self.ffn_dense1(x)
        x = self.ffn_dense2(x)
        x = self.ffn_dense3(x)  # shape -> (batch, seq_len, 1)

        return x