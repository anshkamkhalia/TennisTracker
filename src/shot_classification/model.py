import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, LayerNormalization, MultiHeadAttention, Dense, Dropout, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.models import Model


class ShotClassifier(Model):
    def __init__(self, num_classes=4, **kwargs):
        super().__init__(**kwargs)

        self.input_proj = Dense(128, activation="relu")

        self.conv1 = Conv1D(128, 3, padding="same", activation="relu")
        self.conv2 = Conv1D(128, 5, padding="same", activation="relu")

        self.bn1 = BatchNormalization()

        self.bilstm = Bidirectional(
            LSTM(128, return_sequences=True)
        )

        self.attn = MultiHeadAttention(
            num_heads=4,
            key_dim=64
        )

        self.norm1 = LayerNormalization()

        self.global_pool = GlobalAveragePooling1D()

        self.fc1 = Dense(256, activation="relu")
        self.dropout = Dropout(0.4)

        self.out = Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.input_proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.bn1(x, training=training)

        x = self.bilstm(x)

        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)

        x = self.global_pool(x)

        x = self.fc1(x)
        x = self.dropout(x, training=training)

        return self.out(x)