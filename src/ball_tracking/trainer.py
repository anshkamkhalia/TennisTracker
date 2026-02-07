# trains and exports the final tracker using disk-backed generators
# this version avoids loading the full dataset into ram

from model import TrackNet
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
from tensorflow.keras.utils import Sequence
import numpy as np

class TrackNetDataLoader(Sequence):
    def __init__(self, x_files, y_files, shuffle=True):
        self.x_files = x_files
        self.y_files = y_files
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_files))
        self.on_epoch_end()

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x = np.load(self.x_files[idx])
        y = np.load(self.y_files[idx])

        # cast for mixed precision safety
        x = x.astype("float32")
        y = y.astype("float32")

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# enable mixed precision to reduce vram usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")

root = "src/ball_tracking/ball_tracking_data/labels"

img_h, img_w = 720, 1280
epochs = 100

# list all files in the labels directory
all_files = os.listdir(root)

# separate x and y batch files
x_files = sorted([
    os.path.join(root, f)
    for f in all_files
    if f.startswith("X_train_batch") and f.endswith(".npy")
])

y_files = sorted([
    os.path.join(root, f)
    for f in all_files
    if f.startswith("y_train_batch") and f.endswith(".npy")
])

# sanity check to ensure alignment
assert len(x_files) == len(y_files), "x/y batch count mismatch"

num_batches = len(x_files)

# split by batch count instead of sample count
val_split = 0.1
val_batches = int(num_batches * val_split)

X_val_files = x_files[:val_batches]
y_val_files = y_files[:val_batches]

X_train_files = x_files[val_batches:]
y_train_files = y_files[val_batches:]

tracknet = TrackNet()

train_loader = TrackNetDataLoader(
    X_train_files,
    y_train_files,
    shuffle=True
)

val_loader = TrackNetDataLoader(
    X_val_files,
    y_val_files,
    shuffle=False
)

tracknet.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy"
)

tracknet.fit(
    train_loader,
    validation_data=val_loader,
    epochs=epochs,
    callbacks=[
        ModelCheckpoint("serialized_models/tracknet_best.keras", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=5, monitor="val_loss"),
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    ]
)