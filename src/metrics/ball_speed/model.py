# lightweight lstm to classify contact windows

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, BatchNormalization, Layer
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_layer")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation="tanh")

    def call(self, x, mask=None):
        score = self.W(x)  # (batch, time, 1)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * x, axis=1)

@register_keras_serializable(package="custom_model")
class ContactDetector(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm1 = Bidirectional(LSTM(32, return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(64, return_sequences=True))
        self.attn = Attention()
        self.bn = BatchNormalization()
        self.d1 = Dense(64, activation="relu")
        self.d2 = Dense(128, activation="relu")
        self.out = Dense(1, activation="sigmoid")  # window-level

    def call(self, x, training=False):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.attn(x)
        x = self.bn(x, training=training)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
