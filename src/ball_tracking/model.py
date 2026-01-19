# using a more powerful model (access to colab gpu) for ball detection
# DEPRECATED - not powerful enough hardware, yolo is a better option

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout, LSTM, Layer, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_layer")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation="tanh")

    def call(self, inputs, mask=None):
        score = self.W(inputs)  # (batch, timesteps, 1)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)  # convert bool -> float
            score -= (1.0 - mask[:, :, tf.newaxis]) * 1e9  # large negative for masked
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

@register_keras_serializable(package="custom_model")
class BallTracker(Model):

    """
    a spatiotemporal model to track tennis balls across frames
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from model class

        # architecture

        # conv layers
        self.conv1 = TimeDistributed(Conv2D(16, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-3), strides=2))
        self.bn1 = TimeDistributed(BatchNormalization())

        self.conv2 = TimeDistributed(Conv2D(32, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4), strides=2))
        self.bn2 = TimeDistributed(BatchNormalization())

        self.conv3 = TimeDistributed(Conv2D(64, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4)))
        self.bn3 = TimeDistributed(BatchNormalization())

        self.conv4 = TimeDistributed(Conv2D(128, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4)))
        self.bn4 = TimeDistributed(BatchNormalization())

        self.gap = TimeDistributed(GlobalAveragePooling2D())
        self.dp1 = Dropout(0.25)

        # lstm layers
        self.lstm1 = LSTM(64, activation="tanh", return_sequences=True, kernel_regularizer=l2(1e-4))
        self.attention = Attention()

        self.dense1 = Dense(32, activation="relu", kernel_regularizer=l2(1e-3))
        self.dense2 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))
        self.dense3 = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))
    
        self.out = Dense(4, activation="linear", dtype="float32")
    
    def call(self, x, training=False):
        x = tf.cast(x, tf.float32) / 255.0 # normalize
        # conv forward
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.conv4(x)
        x = self.bn4(x, training=training)

        x = self.gap(x)
        x = self.dp1(x, training=training)

        x = self.lstm1(x)        
        x = self.attention(x)     

        # dense forward
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        # output
        return self.out(x)
    
