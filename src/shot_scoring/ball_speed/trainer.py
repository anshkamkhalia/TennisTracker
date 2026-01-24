# trains an exports the final model

from model import ContactDetector
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
import numpy as np
import os
import gc
from tqdm import tqdm
from tensorflow.keras.saving import register_keras_serializable

root = "src/shot_scoring/ball_speed/labels"

X_train_5 = np.load(os.path.join(root, "videoplayback5_X.npy"))
y_train_5 = np.load(os.path.join(root, "videoplayback5_y.npy"))

X_train_6 = np.load(os.path.join(root, "videoplayback6_X.npy"))
y_train_6 = np.load(os.path.join(root, "videoplayback6_y.npy"))

X_train_8 = np.load(os.path.join(root, "videoplayback8_X.npy"))
y_train_8 = np.load(os.path.join(root, "videoplayback8_y.npy"))

# combine X
X_train_all = np.concatenate([X_train_5, X_train_6, X_train_8], axis=0)

# combine y
y_train_all = np.concatenate([y_train_5, y_train_6, y_train_8], axis=0)
indices = np.arange(len(X_train_all))
np.random.shuffle(indices)

X_train = X_train_all[indices]
y_train = y_train_all[indices]

model = ContactDetector()

model_checkpoint = ModelCheckpoint(
    "serialized_models/contact_detector.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.75,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

@register_keras_serializable(package="custom_loss")
def binary_focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    # clip predictions for numerical stability
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # compute BCE
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # compute focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = alpha * tf.pow(1 - p_t, gamma)

    return tf.reduce_mean(focal_factor * bce)

model.compile(
    optimizer=Adam(1e-3),
    loss=binary_focal_loss,
    metrics=["mse", "mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    callbacks=[reduce_lr, early_stopping, model_checkpoint]
)
