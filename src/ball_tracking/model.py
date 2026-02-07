# using a more powerful model (access to colab gpu) for ball detection

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_model")
class TrackNet(Model):

    """
    a spatiotemporal model to track tennis balls across frames
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs) # inherit from model class

        # architecture for TrackNet

        self.layer1 = Conv2D(64, (3,3), activation="relu", padding="same")
        self.layer2 = BatchNormalization()
        self.layer3 = MaxPooling2D()
        self.layer4 = Conv2D(128, (3,3), activation="relu", padding="same")
        self.layer5 = BatchNormalization()
        self.layer6 = MaxPooling2D()
        self.layer7 = Conv2D(256, (3,3), activation="relu", padding="same")
        self.layer8 = BatchNormalization()
        self.layer9 = MaxPooling2D()
        self.layer10 = Conv2D(512, (3,3), activation="relu", padding="same")
        self.layer11 = BatchNormalization()
        self.layer12 = UpSampling2D()
        self.layer13 = Conv2D(256, (3,3), activation="relu", padding="same")
        self.layer14 = BatchNormalization()
        self.layer15 = UpSampling2D()
        self.layer16 = Conv2D(128, (3,3), activation="relu", padding="same")
        self.layer17 = BatchNormalization()
        self.layer18 = UpSampling2D()
        self.layer19 = Conv2D(64, (3,3), activation="relu", padding="same")
        self.layer20 = BatchNormalization()
        self.layer21 = Conv2D(256, (3,3), activation="relu", padding="same")
        self.layer22 = Conv2D(1, (1,1), activation="sigmoid")

    def call(self, x, training=False):
        # encoder
        x = self.layer1(x)
        x = self.layer2(x, training=training)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(x, training=training)
        x = self.layer6(x)

        x = self.layer7(x)
        x = self.layer8(x, training=training)
        x = self.layer9(x)

        x = self.layer10(x)
        x = self.layer11(x, training=training)

        # decoder
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x, training=training)

        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x, training=training)

        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x, training=training)

        x = self.layer21(x)
        x = self.layer22(x)

        return x
