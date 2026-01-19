# trains and exports the final tracker using disk-backed generators
# this version avoids loading the full dataset into ram

from model import BallTracker, Attention
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

# enable mixed precision to reduce vram usage
tf.keras.mixed_precision.set_global_policy("mixed_float16")

root = "src/ball_tracking/videos/labels"

sequence_len = 10
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

x_val_files = x_files[:val_batches]
y_val_files = y_files[:val_batches]

x_train_files = x_files[val_batches:]
y_train_files = y_files[val_batches:]

print(f"train batches: {len(x_train_files)}")
print(f"val batches: {len(x_val_files)}")

def batch_file_generator(x_paths, y_paths, shuffle=True):
    # create indices so we can shuffle without touching disk order
    indices = np.arange(len(x_paths))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            # load a single batch pair into memory
            x = np.load(x_paths[i])
            y = np.load(y_paths[i])

            yield x, y

            # aggressively free ram after yielding
            del x, y
            gc.collect()

# explicitly define shapes so tensorflow does not guess wrong
output_signature = (
    tf.TensorSpec(
        shape=(None, sequence_len, img_h, img_w, 3),
        dtype=tf.uint8
    ),
    tf.TensorSpec(
        shape=(None, 4),
        dtype=tf.float32
    )
)

options = tf.data.Options()
options.autotune.enabled = False

# training dataset streams batches from disk
train_dataset = tf.data.Dataset.from_generator(
    lambda: batch_file_generator(x_train_files, y_train_files, shuffle=True),
    output_signature=output_signature
).prefetch(1)

# validation dataset does not shuffle
val_dataset = tf.data.Dataset.from_generator(
    lambda: batch_file_generator(x_val_files, y_val_files, shuffle=False),
    output_signature=output_signature
).prefetch(1)

train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

ball_tracker = BallTracker()

optimizer = Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

ball_tracker.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["mae"]
)

model_checkpoint = ModelCheckpoint(
    "serialized_models/ball_tracker.keras",
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
    patience=15,
    restore_best_weights=True
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# manually attach callbacks for custom loop
for cb in callbacks:
    cb.set_model(ball_tracker)
    cb.on_train_begin()

train_loss = tf.keras.metrics.Mean(name="train_loss")

for epoch in range(epochs):

    print(f"\nepoch {epoch + 1}/{epochs}")
    train_loss.reset_state()

    for cb in callbacks:
        cb.on_epoch_begin(epoch)

    # training pass (one disk batch per step)
    for step, (x_batch, y_batch) in enumerate(
        tqdm(
            train_dataset.take(len(x_train_files)),
            desc="training",
            ncols=100
        )
    ):
        for cb in callbacks:
            cb.on_train_batch_begin(step)

        with tf.GradientTape() as tape:
            preds = ball_tracker(x_batch, training=True)
            loss = loss_fn(y_batch, preds)

        grads = tape.gradient(loss, ball_tracker.trainable_variables)
        optimizer.apply_gradients(zip(grads, ball_tracker.trainable_variables))

        train_loss.update_state(loss_fn(y_batch, preds))

        if step % 5 == 0:
            tqdm.write(f"step {step + 1} | loss: {loss.numpy():.4f}")

        for cb in callbacks:
            cb.on_train_batch_end(step, logs={"loss": loss.numpy()})

        # free memory aggressively
        del x_batch, y_batch, preds, grads
        gc.collect()

    # validation pass
    val_losses = []

    for x_batch, y_batch in tqdm(
        val_dataset.take(len(x_val_files)),
        desc="validation",
        ncols=100
    ):
        preds = ball_tracker(x_batch, training=False)
        loss = loss_fn(y_batch, preds)
        val_losses.append(loss.numpy())

        del x_batch, y_batch, preds
        gc.collect()

    val_loss = float(np.mean(val_losses))
    print(f"val loss: {val_loss:.4f}")

    for cb in callbacks:
        cb.on_epoch_end(
            epoch,
            logs={
                "loss": train_loss.result().numpy(),
                "val_loss": val_loss
            }
        )

    if ball_tracker.stop_training:
      break

for cb in callbacks:
    cb.on_train_end()
