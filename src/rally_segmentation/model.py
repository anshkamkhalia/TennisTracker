import tensorflow as tf
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = layers.Embedding(seq_len, d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.pos_emb(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    # def compute_mask(self, inputs, mask=None):
    #     return mask

    def call(self, x, training=False):
        attn_out = self.attn(
            query=x,
            value=x,
            key=x,
            training=training
        )

        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out, training=training)

        return self.norm2(x + ffn_out)


class ContactTransformer(tf.keras.Model):
    def __init__(
        self,
        seq_len=15,
        num_features=18,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = layers.Dense(d_model)

        self.pos_encoding = PositionalEncoding(seq_len, d_model)

        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.pool = layers.GlobalAveragePooling1D()

        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, x, training=False):
        seq_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)

        x = self.input_proj(x)

        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.pool(x, mask=seq_mask)

        return self.classifier(x, training=training)