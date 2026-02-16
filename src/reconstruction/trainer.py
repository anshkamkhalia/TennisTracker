# trains a reconstruction network

import tensorflow as tf
import numpy as np
import pandas as pd
from model import Reconstructor
import gc
import os

tf.keras.mixed_precision.set_dtype_policy('mixed_float16') # set to float 16 instead of float 32
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

reconstructor = Reconstructor() # initialize model
reconstructor.build(None, 60, 2) # batch size, trajectories, features
data_root_dir = "src/reconstruction/synthetic_data"

def synthetic_data_gen(feature_path, label_path):
    """used to lazily load data"""
    # keep in float32
    X_train = np.load(feature_path, dtype=np.float32)
    y_train = np.loadr(label_path=np.float32)

    yield X_train, y_train

def multi_game_generator(X_list, y_list):
    """for loading multiple datasets"""
    for x, y in zip(X_list, y_list):
        X_path = os.path.join(data_root_dir, x) # reference root dir
        y_path = os.path.join(data_root_dir, y) # reference root dir
        yield from synthetic_data_gen(X_path, y_path)


# model setup
optimizer = tf.keras.optimizers.Adam(2e-4)
loss_fn = tf.keras.losses.MeanSquaredError()
EPOCHS = 100
BATCH_SIZE = 32
SHUFFLE = 1000
PATIENCE = 20

train_files_features = [f"X_train_chunk{i}" for i in range(1,45)]
test_files_features = [f"X_train_chunk{i}" for i in range(45,51)]
train_files_labels = [f"y_train_chunk{i}" for i in range(1,45)]
test_files_labels = [f"y_train_chunk{i}" for i in range(45,51)]

@tf.function
def train_step(x,y):

    """uses tf.function to optimize training speed"""

    vars = reconstructor.trainable_variables
    with tf.GradientTape() as tape:
        preds = reconstructor(x, training=True)
        preds = tf.cast(preds, tf.float32)

        # loss
        y = tf.cast(y, tf.float32)
        loss = loss_fn(y, preds)

    grads = tape.gradient(loss, vars) # calculate gradients
    optimizer.apply_gradients(zip(grads, vars))# apply gradients
    return loss

# model checkpointing params
best_global_test_loss = float("inf") # for model checkpointing
best_test_loss = float("inf") # also for model checkpointing
global_no_improve_count = 0 # for early stopping 
no_improve_count = 0

# create training dataset
dataset = (
    tf.data.Dataset.from_generator(

        lambda: multi_game_generator(train_files_features, train_files_labels), # generator function
        output_signature=(
            tf.TensorSpec(shape=(None, 60, 2), dtype=tf.float16), # inputs
            tf.TensorSpec(shape=(None, 60, 1), dtype=tf.float16), # outputs
        )
    )
    .shuffle(SHUFFLE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("train dataset created")

# create val dataset
test_dataset = (
    tf.data.Dataset.from_generator(

        lambda: multi_game_generator(test_files_features, test_files_labels), # generator function
        output_signature=(
            tf.TensorSpec(shape=(None, 60, 2), dtype=tf.float16), # inputs
            tf.TensorSpec(shape=(None, 60, 1), dtype=tf.float16), # outputs
        )
    )
    .shuffle(SHUFFLE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("test dataset created")

# training loop
for epoch in range(EPOCHS):
    print(f"epoch {epoch+1}\n")

    train_loss = 0.0
    train_samples = 0 
    train_iter = iter(dataset) # avoid generator exhaustion

    # train model
    for x,y in train_iter:

        loss = train_step(x,y) # defined above
        train_loss += loss.numpy() * x.shape[0]
        train_samples += x.shape[0] # n samples per batch

        del x,y # instant gc

    if train_samples > 0 :
        train_loss /= train_samples # average
    else:
        print("no train samples this epoch")
        continue

    # evaluate model
    test_loss = 0.0
    test_samples = 0
    test_iter = iter(test_dataset)

    for step, (x,y) in enumerate(test_iter):
        preds = reconstructor(x, training=False)
        preds = tf.cast(preds, tf.float32)
        y = tf.cast(y, tf.float32)
        loss = loss_fn(y, preds)
        test_loss += loss.numpy() * x.shape[0]
        test_samples += x.shape[0]
    
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
        reconstructor.save("serialized_models/reconstructor.keras")  # global best
        print(f"saved GLOBAL best model, test_loss: {test_loss}")
    else:
        no_improve_count += 1
        global_no_improve_count += 1

    if no_improve_count > PATIENCE:
        break

    gc.collect()