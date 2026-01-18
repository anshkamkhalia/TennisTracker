# using a more powerful model (access to colab gpu) for ball detection

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout, LSTM, Layer
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
    a neural net to track 14 different keypoints on tennis courts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from model class

        # architecture

        # conv layers
        self.conv1 = Conv2D(32, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-3), strides=2)
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(64, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4), strides=2)
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(128, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4))
        self.bn3 = BatchNormalization()

        self.conv4 = Conv2D(256, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4))
        self.bn4 = BatchNormalization()

        self.gap = GlobalAveragePooling2D()
        self.dp1 = Dropout(0.25)

        # lstm layers
        self.lstm1 = LSTM(64, activation="tanh", return_sequences=True, kernel_regularizer=l2(1e-4))
        self.attention = Attention()
        self.lstm2 = LSTM(128, activation="tanh", return_sequences=True, kernel_regularizer=l2(1e-4))

        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2(1e-3))
        self.dense2 = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))
        self.dense3 = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))

        self.dp2 = Dropout(0.3)
        self.dense4 = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))
    
        self.out = Dense(6, activation="linear") # 6 values (x1, y1, x2, y2, cx, cy)
    
    def call(self, x, training=False):
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

        # reshape for LSTM: (batch, timesteps=1, features)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm1(x)
        x = self.attention(x)
        x = self.lstm2(x)

        # dense forward
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dp2(x, training=training)
        x = self.dense4(x)

        # output
        return self.out(x)
    
