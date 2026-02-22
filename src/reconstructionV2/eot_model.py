# lstm + attention

import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable(package="custom_layer")
class Attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.W = None
        self.b = None

    def build(self, input_shape):
        # input_shape = (batch_size, timesteps, features)
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True,
                                 name="attn_W")
        self.b = self.add_weight(shape=(input_shape[1], 1), # timesteps bias
                                 initializer="zeros",
                                 trainable=True,
                                 name="attn_b")
        super().build(input_shape)

    def call(self, x):
        # compute scores
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        # attention weights
        attn_weights = tf.nn.softmax(score, axis=1) # softmax over timesteps

        # weighted sum
        weighted_sum = tf.reduce_sum(x * attn_weights, axis=1)
        weighted_sum = tf.expand_dims(weighted_sum, axis=1)  
        return weighted_sum

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

@tf.keras.utils.register_keras_serializable(package="custom_model")
class EoTNetwork(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # lstm block 1
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128, 
                activation="tanh", 
                kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4), 
                return_sequences=True
                )
        )
        self.bn1 = tf.keras.layers.BatchNormalization() 

        # lstm block 2
        self.lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256, 
                activation="tanh", 
                kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4), 
                return_sequences=False # flatten for dense
                )
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        # fully connected layers
        self.dense1 = tf.keras.layers.Dense(128, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(256, activation="gelu")
        self.out = tf.keras.layers.Dense(2, activation=None)

    def call(self, x):

        x = self.lstm1(x)
        x = self.bn1(x)

        x = self.lstm2(x)
        x = self.bn2(x)

        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        out = out * 3 # 3 meters is max height; denormalize
        return out