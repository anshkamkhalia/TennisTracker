import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LayerNormalization, MaxPooling2D, UpSampling2D, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom_model")
class TrackNet(Model):

    """note: uses a slightly tweaked version for lesser memory usage"""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        # encoder
        self.layer1 = SeparableConv2D(64, (3,3), activation="gelu", padding="same")
        self.layer2 = LayerNormalization()
        self.layer3 = MaxPooling2D(pool_size=2, strides=1)

        self.layer4 = SeparableConv2D(128, (3,3), activation="gelu", padding="same")
        self.layer5 = LayerNormalization()
        self.layer6 = MaxPooling2D(pool_size=2, strides=1)

        self.layer7 = SeparableConv2D(256, (3,3), activation="gelu", padding="same")
        self.layer8 = LayerNormalization()
        self.layer9 = MaxPooling2D(pool_size=2, strides=1)

        self.layer10 = SeparableConv2D(256, (3,3), activation="gelu", padding="same")
        self.layer11 = LayerNormalization()

        # decoder
        self.layer12 = UpSampling2D()
        self.layer13 = SeparableConv2D(192, (3,3), activation="gelu", padding="same")
        self.layer14 = LayerNormalization()

        self.layer15 = UpSampling2D()
        self.layer16 = SeparableConv2D(128, (3,3), activation="gelu", padding="same")
        self.layer17 = LayerNormalization()

        self.layer18 = UpSampling2D()
        self.layer19 = SeparableConv2D(64, (3,3), activation="relu", padding="same")
        self.layer20 = LayerNormalization()

        self.layer21 = SeparableConv2D(32, (3,3), activation="relu", padding="same")
        self.layer22 = SeparableConv2D(1, (1,1), activation="sigmoid")

    def call(self, x, training=False):
        # encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = self.layer10(x)
        x = self.layer11(x)

        # decoder
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)

        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)

        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)

        x = self.layer21(x)
        x = self.layer22(x)

        return x
