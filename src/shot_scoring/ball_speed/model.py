from tensorflow.keras.layers import LSTM, Dense, Bidirectional, GlobalAveragePooling1D, BatchNormalization, Layer
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_layer")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation="tanh")

    def call(self, inputs, mask=None):
        score = self.W(inputs)  # (batch, timesteps, 1)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            score -= (1.0 - mask[:, :, tf.newaxis]) * 1e9
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

@register_keras_serializable(package="custom_model")
class ContactDetector(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm1 = Bidirectional(LSTM(32, activation="tanh", return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(64, activation="tanh", return_sequences=True))
        self.attention = Attention()
        self.bn = BatchNormalization()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(128, activation="relu")
        self.out = Dense(21, activation="sigmoid")  # one per timestep

    def call(self, x, training=False):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.attention(x)  # outputs (batch, features)
        x = self.bn(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)  # final shape: (batch, 21)
