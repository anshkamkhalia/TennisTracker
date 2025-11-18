# creates a deep learning model using tensorflow

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dropout, Layer, TimeDistributed, Conv3D
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# attention layer
@register_keras_serializable(package="custom_layer") # to save along with model
class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(1, activation='tanh')
    
    def call(self, inputs):
        # inputs: (batch_size, timesteps, features)
        score = self.W(inputs)  # (batch_size, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)  # attention weights
        context = tf.reduce_sum(weights * inputs, axis=1)  # weighted sum
        return context

@register_keras_serializable(package="custom_model") # to be saved/loaded with keras later
class TennisTracker(Model):

    """

    a 3dcnn + lstm based neural network to classify tennis shots
    
    layers (not in this order):
        - dense
        - lstm 
        - batchnormalization
        - dropout
        - custom attention
        - conv3d

    loss - binary_cross_entropy (subject to change -> sparse)
    
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs) # inheritance from Model class

        # model architecture

        # timedistributed dense and dropout
        self.td_dense1 = Dense(512, activation="relu")

        self.dropout = Dropout(0.25) # prevent overfitting

        self.td_dense2 = Dense(256, activation="relu")
        self.td_dense3 = Dense(128, activation="relu")

        # lstm
        self.lstm1 = LSTM(64, activation="tanh")
        self.lstm2 = LSTM(32, activation="tanh")

        # conv3d
        self.conv3d_1 = Conv3D(64, (3,3,3), padding='same', # l2 for generalizing
                             kernel_regularizer=l2(1e-4))
        self.conv3d_2 = Conv3D(32, (3,3,3), padding='same', # 3x3x3 (conv3d_1) -> 3x3x3 (conv3d_1) stacks well
                             kernel_regularizer=l2(1e-4)) 
        
        # attention
        self.attention = Attention() # emphasizes certain important frames

        self.bn = BatchNormalization()  # normalize final vector

        self.out = Dense(1, activation="sigmoid") # binary classification, 2 classes, sigmoid

    def call(self, x):

        x = self.td_dense1(x)

        x = self.dropout(x)

        x = self.td_dense2(x)
        x = self.td_dense3(x)

        x = self.lstm1(x)
        x = self.lstm2(x)

        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        x = self.attention(x)

        x = self.bn(x)

        out = self.out(x)

        return out