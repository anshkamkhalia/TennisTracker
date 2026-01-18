# trains and exports the final tracker

from model import BallTracker, Attention
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from tqdm import tqdm

tf.keras.mixed_precision.set_global_policy('mixed_float16')
root = "src/ball_tracking/videos/labels"

# list of video indices to load
video_ids = [2, 3, 4, 5, 6, 8]

X_train_list = []
y_train_list = []

for vid in video_ids:
    X_train_list.append(np.load(os.path.join(root, f"X_train_videoplayback{vid}.npy")))
    y_train_list.append(np.load(os.path.join(root, f"y_train_videoplayback{vid}.npy")))

# concatenate all arrays into single X and y
X_train = np.vstack(X_train_list)  # if each X_train_list[i] = (num_sequences, SEQUENCE_LEN, H, W, 3)
y_train = np.vstack(y_train_list)

# callbacks

# model checkpoint - saves best model during training
model_checkpoint = ModelCheckpoint(
    'src/ball_tracking/saved_model/ball_tracker.keras',         # file to save
    monitor='val_loss',      # what to monitor
    save_best_only=True,     # only save when its the best
    verbose=1
)

# reduce lr - reduces lr by a dampening factor if loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.75,              
    patience=10,              # wait this many epochs before reducing
    min_lr=1e-6,             # dont go below this LR
    verbose=1 
)

# early stopping - stops if model has not been improving
early_stopping = EarlyStopping(
    monitor='val_loss',      # what to monitor
    patience=15,             # how many epochs to wait before stopping
    restore_best_weights=True
)

BATCH_SIZE = 16
EPOCHS = 100

# optimizer and loss
optimizer = Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

# metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")

ball_tracker = BallTracker()

# compile model
ball_tracker.compile(
    optimizer=optimizer,
    loss=loss_fn, # multiple labels
    metrics=['mae']
)

val_split = 0.1
val_size = int(len(X_train) * val_split)

X_test = X_train[:val_size]
y_test = y_train[:val_size]

X_train = X_train[val_size:]
y_train = y_train[val_size:]

# custom training loop
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=500).batch(batch_size=BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(batch_size=BATCH_SIZE)

callbacks = early_stopping, reduce_lr, model_checkpoint

# tell each callback the model
for cb in callbacks:
    cb.set_model(ball_tracker)
    cb.on_train_begin()

for epoch in range(EPOCHS):
    
    for cb in callbacks:
        cb.on_epoch_begin(epoch)

    print(f"epoch: {epoch+1}/{EPOCHS}")

    train_loss.reset_states() # reset metric
    val_losses = []            # reset val losses

    for step, (X_batch, y_batch) in enumerate(tqdm(dataset, desc="training", ncols=100)):
        for cb in callbacks:
            cb.on_train_batch_begin(step)

        with tf.GradientTape() as tape:
            y_pred = ball_tracker(X_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, ball_tracker.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ball_tracker.trainable_variables))

        train_loss.update_state(loss)

        if step % 10 == 0:
            tqdm.write(f"step {step+1} - batch loss: {loss.numpy():.4f}")

        for cb in callbacks:
            cb.on_train_batch_end(step, logs={"loss": loss.numpy()})

    # validation
    val_loss = 0
    val_steps = 0
    for X_batch, y_batch in tqdm(val_dataset, desc="validation", ncols=100):
        preds = ball_tracker(X_batch, training=False)
        loss = loss_fn(y_batch, preds)
        val_losses.append(loss.numpy())
        val_loss += loss.numpy()
        val_steps += 1
    val_loss /= val_steps

    print(f"val loss for epoch {epoch+1}: {np.mean(val_losses)}")
    for cb in callbacks:
        cb.on_epoch_end(epoch, logs={"loss": train_loss.result().numpy(), "val_loss": val_loss})

    # print(f"\n\nepoch {epoch+1}\nloss: {train_loss.result().numpy():.4f}\nval_loss: {np.mean(val_losses)}\n\n")

for cb in callbacks:
    cb.on_train_end()