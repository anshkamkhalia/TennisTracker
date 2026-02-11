# creates a model using the subclassing api of tensorflow

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, LayerNormalization, LSTM, Dropout, Bidirectional, TimeDistributed, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

# attention layer
@register_keras_serializable(package="custom_layer")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation="tanh")

    def call(self, inputs, mask=None):
        score = self.W(inputs)  # (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context
    
@register_keras_serializable(package="custom_layer")
class SequenceAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation="tanh")
        self.supports_masking = True

    def call(self, inputs, mask=None):
        score = self.W(inputs)  # (batch, T, 1)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            score -= (1.0 - mask[:, :, None]) * 1e9

        weights = tf.nn.softmax(score, axis=1)
        return inputs * weights  # (batch, T, F)

@register_keras_serializable(package="custom_model")
class ShotClassifier(Model):
    
    """
    a neural net that can classify the type of shot given the keypoints
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from Model() class

        # create architecture

        # self.mask = Masking(mask_value=0.0) # add masking to ignore padded values
        # self.attention1 = SequenceAttention()
        self.bn_lstm1 = Bidirectional(LSTM(512, return_sequences=True, recurrent_regularizer=l2(1e-4), activation="tanh"))
        self.dropout = Dropout(0.3)

        self.bn_lstm2 = Bidirectional(LSTM(256, return_sequences=True, recurrent_regularizer=l2(1e-4), activation="tanh"))
        self.dropout2 = Dropout(0.3)

        self.td_dense = TimeDistributed(Dense(256, activation="relu", kernel_regularizer=l2(1e-4)))

        self.bn_lstm3 = Bidirectional(LSTM(128, return_sequences=True, recurrent_regularizer=l2(1e-4), activation="tanh"))
        self.attention2 = Attention()

        self.dense1 = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))
        self.layer_norm = LayerNormalization()

        self.dropout3 = Dropout(0.4)
        self.dense3 = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))
        self.dense4 = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))

        self.out = Dense(4, activation="softmax") # subject to change (n_classes)

    def call(self, inputs, training=False):
        # inputs: (batch, timesteps, features)

        # x = self.attention1(x)
        x = self.bn_lstm1(inputs)
        x = self.dropout(x, training=training)

        x = self.bn_lstm2(x)
        x = self.dropout2(x)

        x = self.td_dense(x)

        x = self.bn_lstm3(x)
        x = self.attention2(x)  # (batch, features)

        x = self.dense1(x)
        x = self.layer_norm(x, training=training)

        x = self.dropout3(x, training=training)
        x = self.dense3(x)

        return self.out(x)