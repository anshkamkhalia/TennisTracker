# uses on the fly preprocessing to train a model
# reduces OOM, blobs of .npy files, and batching issues
# uses lazy loading with a tf.data.Data object and generators

import tensorflow as tf
import cv2 as cv
import numpy as np
import pandas as pd
from model import TrackNet
import gc
import os
from typing import Generator

tf.keras.mixed_precision.set_dtype_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tracknet = TrackNet()
dummy_input = tf.zeros((1, 360, 640, 3), dtype=tf.float32)
_ = tracknet(dummy_input)  # now model builds
data_root_dir = "src/ball_tracking/ball_tracking_data/Dataset"

def make_gaussian_kernel(size=21, sigma=4):
    """builds a small gaussian kernel to avoid creating multiple full heatmaps"""

    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel.astype(np.float16)

GAUSSIAN = make_gaussian_kernel(size=21, sigma=4)
KERNEL_SIZE = GAUSSIAN.shape[0]
HALF = KERNEL_SIZE // 2

def data_generator(game_path: str):
    
    """generator for tf.data.Dataset objects"""

    clips = os.listdir(game_path) # get clip directories
    for clip in clips:

            if clip == ".DS_Store":
                continue

            clip_path = os.path.join(game_path, clip) # get path of current clip

            labels_df = pd.read_csv(os.path.join(clip_path, "Label.csv")) # load labels

            # extract relevant information
            xs = labels_df["x-coordinate"].values
            ys = labels_df["y-coordinate"].values
            img_filenames = labels_df["file name"].tolist()

            centers = np.array(list(zip(xs, ys))) # array of ball centers for heatmap

            for filename, (cx, cy) in zip(img_filenames, centers):
                
                # load image
                img = cv.imread(os.path.join(clip_path, filename))
                img = cv.resize(img, (640, 360)) # resize image
                img = img.astype(np.float16) / 255.0 # normalize

                heatmap = np.zeros((360, 640), dtype=np.float16) # create heatmap

                if cx >= 0 and cy >= 0:

                    cx = int(cx)
                    cy = int(cy)

                    x1 = max(cx - HALF, 0)
                    y1 = max(cy - HALF, 0)
                    x2 = min(cx + HALF + 1, 640)
                    y2 = min(cy + HALF + 1, 360)

                    kx1 = max(HALF - cx, 0)
                    ky1 = max(HALF - cy, 0)
                    kx2 = kx1 + (x2 - x1)
                    ky2 = ky1 + (y2 - y1)

                    if kx2 > kx1 and ky2 > ky1:  # ensures non-empty slice
                        heatmap[y1:y2, x1:x2] = GAUSSIAN[ky1:ky2, kx1:kx2]

                yield img, heatmap[..., None]

def multi_game_generator(game_list):
    """for loading multiple games"""
    for game in game_list:
        game_path = os.path.join(data_root_dir, game)
        yield from data_generator(game_path)

# time to train model
# setup
optimizer = tf.keras.optimizers.Adam(2e-4)
loss_fn = tf.keras.losses.BinaryCrossentropy()
EPOCHS = 25
BATCH_SIZE = 4
SHUFFLE = 1000

train_games = [f"game{i}" for i in range(1, 9)] # 8 games for training
val_games = [f"game{i}" for i in range(9, 11)] # 2 games for testing
patience = 20 # n epochs of no improvement before stopping

@tf.function
def train_step(x, y):

    """uses @tf.function for optimized ops"""

    vars = tracknet.trainable_variables

    with tf.GradientTape() as tape:
        preds = tracknet(x, training=True)
        preds = tf.cast(preds, tf.float32) # cast to float32 to avoid frying gradients
        # loss
        y = tf.cast(y, tf.float32)
        loss = loss_fn(y, preds)

    grads = tape.gradient(loss, vars) # calculate gradients
    optimizer.apply_gradients(zip(grads, vars)) # apply gradients
    return loss

# model checkpointing
best_global_test_loss = float("inf") # for model checkpointing
best_test_loss = float("inf") # also for model checkpointing
global_no_improve_count = 0 # for early stopping 
no_improve_count = 0

# create training dataset
dataset = (
    tf.data.Dataset.from_generator(

        lambda: multi_game_generator(train_games), # generator function
        output_signature=(
            tf.TensorSpec(shape=(360, 640, 3), dtype=tf.float16), # inputs
            tf.TensorSpec(shape=(360, 640, 1), dtype=tf.float16) # outputs
        )

    )
    .shuffle(SHUFFLE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("defined train dataset")

# creat validation set
test_dataset = (
    tf.data.Dataset.from_generator(

        lambda: multi_game_generator(val_games),
        output_signature=(
            tf.TensorSpec(shape=(360, 640, 3), dtype=tf.float16), # inputs
            tf.TensorSpec(shape=(360, 640, 1), dtype=tf.float16) # outputs
        )

    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)

)

print("val dataset defined")

# main training loop
for epoch in range(EPOCHS):
    print(f"epoch {epoch+1}\n")
    
    train_loss = 0.0
    train_samples = 0

    train_iter = iter(dataset) # avoid generator exhaustion

    for x,y in train_iter:

        loss = train_step(x,y) # defined above
        train_loss += loss.numpy() * x.shape[0]
        train_samples += x.shape[0] # n samples per batch
        print(f"train_samples:{train_samples}")

        del x,y # instant gc

    if train_samples > 0 :
        train_loss /= train_samples # average
    else:
        print("no train samples this epoch")
        continue

    test_loss = 0.0
    test_samples = 0
    
    test_iter = iter(test_dataset)

    for step, (x,y) in enumerate(test_iter):
        preds = tracknet(x, training=False)
        preds = tf.cast(preds, tf.float32)
        y = tf.cast(y, tf.float32)
        loss = loss_fn(y, preds)
        test_loss += loss.numpy() * x.shape[0]
        test_samples += x.shape[0]

        # store val predictions
        if not os.path.exists("tracknet_results"):
            os.makedirs("tracknet_results", exist_ok=True)
        else: pass

        if step % 200 == 0:
            os.makedirs(f"tracknet_results/epoch{epoch+1}", exist_ok=True)
            preds_to_save = np.array(preds[0, ..., 0] * 255).astype(np.uint8)  # remove batch and channel dim
            true_to_save = np.array(y[0, ..., 0] * 255).astype(np.uint8)
            # apply a colormap for visualization
            pred_colored = cv.applyColorMap(preds_to_save, cv.COLORMAP_JET)
            true_colored = cv.applyColorMap(true_to_save, cv.COLORMAP_JET)

            cv.imwrite(os.path.join(f"tracknet_results/epoch{epoch+1}", f"test_sample_pred_step{step}.jpeg"), pred_colored)
            cv.imwrite(os.path.join(f"tracknet_results/epoch{epoch+1}", f"test_sample_true_step{step}.jpeg"), true_colored)

            del true_to_save, preds_to_save

    if test_samples > 0:
        test_loss /= test_samples # average over dataset
    else:
        print("no test samples in this epoch")
        continue

    print(f"train_loss: {train_loss} | test_loss {test_loss}") # logging

    # model checkpointing
    if test_loss < best_global_test_loss:
        best_global_test_loss = test_loss
        global_no_improve_count = 0
        tracknet.save("serialized_models/tracknet_best.keras")  # global best
        print(f"saved GLOBAL best model, test_loss: {test_loss}")
    else:
        no_improve_count += 1
        global_no_improve_count += 1

    if no_improve_count > patience:
        break

    gc.collect()