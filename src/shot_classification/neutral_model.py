# creates a model using the subclassing api from tensorflow
# similar to model.py but with small changes

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Conv2D, Flatten, LayerNormalization, Dropout
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
        if mask is not None:
            mask = tf.cast(mask, tf.float32)  # convert bool -> float
            score -= (1.0 - mask[:, :, tf.newaxis]) * 1e9  # large negative for masked
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

@register_keras_serializable(package="custom_model")
class NeutralIdentifier(Model):
    
    """
    a neural net that can classify if a player is neutral or swinging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from Model() class

        # convolutional layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')
        
        # Flatten + normalization + dropout
        self.flatten = Flatten()
        self.layernorm = LayerNormalization()
        self.dropout = Dropout(0.3)
        
        # dense layers
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(64, activation='relu')
        
        # output
        self.out = Dense(1, activation="sigmoid")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)