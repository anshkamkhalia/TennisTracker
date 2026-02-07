# creates a relatively simple model to track keypoints on tennis courts
# DEPRECATED

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_model")
class CourtDetector(Model):

    """
    a neural net to track 14 different keypoints on tennis courts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from model class

        # architecture
        self.conv1 = Conv2D(32, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-3), strides=2)
        self.conv2 = Conv2D(64, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4), strides=2)
        self.conv3 = Conv2D(128, 3, padding='same', activation="relu", kernel_regularizer=l2(1e-4))
        
        self.bn1 = BatchNormalization()
        self.gap = GlobalAveragePooling2D()
        self.dp = Dropout(0.25)
        
        self.dense1 = Dense(256, activation="relu", kernel_regularizer=l2(1e-3))
        self.dense2 = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))
        self.dense3 = Dense(64, activation="relu", kernel_regularizer=l2(1e-4))
    
        self.out = Dense(28, activation="linear") # 28 values -> 14 coordinates (xN, yN)

    
    def call(self, x, training=False):
        # conv layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.conv3(x)
        
        # flatten and dropout
        x = self.gap(x)
        x = self.dp(x, training=training)
        
        # dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        
        # output
        x = self.out(x)
        
        return x
    
