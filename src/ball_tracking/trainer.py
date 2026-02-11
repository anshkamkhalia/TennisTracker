import tensorflow as tf
import numpy as np
import os
import gc
from tqdm import tqdm
from model import TrackNet

tf.keras.mixed_precision.set_global_policy("mixed_float16")

root = "src/ball_tracking/ball_tracking_data/labels"
epochs = 100
lr = 1e-4

all_files = os.listdir(root)
x_files = sorted([os.path.join(root, f) for f in all_files if f.startswith("X_train_batch")])
y_files = sorted([os.path.join(root, f) for f in all_files if f.startswith("y_train_batch")])

assert len(x_files) == len(y_files)

num_batches = len(x_files)
val_split = 0.1
val_batches = int(num_batches * val_split)

x_val = x_files[:val_batches]
y_val = y_files[:val_batches]

x_train = x_files[val_batches:]
y_train = y_files[val_batches:]

def data_gen(x_list, y_list):
    """Yield one batch at a time as float16"""
    for x_path, y_path in zip(x_list, y_list):
        x = np.load(x_path).astype("float16")
        y = np.load(y_path).astype("float16")
        yield x, y

def create_dataset(x_list, y_list):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_gen(x_list, y_list),
        output_signature=(
            tf.TensorSpec(shape=(None, 720, 1280, 3), dtype=tf.float16),
            tf.TensorSpec(shape=(None, 720, 1280, 1), dtype=tf.float16),
        )
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

model = TrackNet()
optimizer = tf.keras.optimizers.Adam(lr)
loss_fn = tf.keras.losses.BinaryCrossentropy()
best_val_loss = float("inf")

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_loss = 0.0
    for x, y in tqdm(train_dataset, total=len(x_train)):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss += loss.numpy()
        del x, y, preds, grads
        gc.collect()

    train_loss /= len(x_train)
    print(f"train loss: {train_loss:.5f}")

    val_loss = 0.0
    for x, y in val_dataset:
        preds = model(x, training=False)
        loss = loss_fn(y, preds)
        val_loss += loss.numpy()

        del x, y, preds
        gc.collect()

    val_loss /= max(1, len(x_val))
    print(f"val loss: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save("serialized_models/tracknet_best.keras")
        print("saved new best model")
