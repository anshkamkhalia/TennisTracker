from model import ContactDetector
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from tensorflow.keras.saving import register_keras_serializable

root = "src/shot_scoring/ball_speed/labels"

Xs, ys = [], []

for name in ["videoplayback5", "videoplayback6", "videoplayback7"]:
    Xs.append(np.load(os.path.join(root, f"{name}_X.npy")))
    ys.append(np.load(os.path.join(root, f"{name}_y.npy")))

X = np.concatenate(Xs)
y = np.concatenate(ys)

idx = np.arange(len(X))
np.random.shuffle(idx)
X, y = X[idx], y[idx]

@register_keras_serializable(package="custom_loss")
def binary_focal_loss(y_true, y_pred):
    gamma_pos = 4.0   # focus HARD on missed contacts
    gamma_neg = 2.0

    alpha_pos = 0.85  # positives matter more
    alpha_neg = 0.15

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # BCE
    bce = -(y_true * tf.math.log(y_pred) +
            (1 - y_true) * tf.math.log(1 - y_pred))

    # p_t
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # class-specific gamma + alpha
    focal = (
        y_true * alpha_pos * tf.pow(1 - p_t, gamma_pos) +
        (1 - y_true) * alpha_neg * tf.pow(1 - p_t, gamma_neg)
    )

    return tf.reduce_mean(focal * bce)

model = ContactDetector()

model.compile(
    optimizer=Adam(1e-3),
    loss=binary_focal_loss,
    metrics=[
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# model.fit(
#     X,
#     y,
#     epochs=100,
#     batch_size=16,
#     validation_split=0.15,
#     callbacks=[
#         ModelCheckpoint("serialized_models/contact_detector.keras", save_best_only=True),
#         ReduceLROnPlateau(patience=10),
#         EarlyStopping(patience=20, restore_best_weights=True),
#     ],
# )
