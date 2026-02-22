# trains EoT model

import tensorflow as tf
from eot_model import EoTNetwork
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# hyperparams
seq_len = 60
batch_size = 64
epochs = 70
learning_rate = 3e-4
data_dir = "src/reconstructionV2/synthetic_data"
strokes = ["groundstroke", "serve", "lob"]
num_batches_per_type = 15
validation_split = 0.1
os.makedirs("src/reconstructionV2/graph_checkpoints", exist_ok=True)

def load_all_batches(data_dir=data_dir, strokes=strokes, num_batches=num_batches_per_type):

    x_list, y_list = [], []

    for stroke in strokes:
        for b in range(1, num_batches + 1):
            x_path = f"{data_dir}/X_train_batch_{b}.npy"
            y_path = f"{data_dir}/y_train_batch_{b}.npy"
            x = np.load(x_path)
            y = np.load(y_path)
            x_list.append(x)
            y_list.append(y)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    print(f"loaded {x_all.shape[0]} samples with shape {x_all.shape[1:]}")
    return x_all, y_all

x_all, y_all = load_all_batches()

# shuffle dataset
idx = np.arange(x_all.shape[0])
np.random.shuffle(idx)
x_all = x_all[idx]
y_all = y_all[idx]

# train split
val_size = int(len(x_all) * validation_split)
x_val = x_all[:val_size]
y_val = y_all[:val_size]
x_train = x_all[val_size:]
y_train = y_all[val_size:]

# convert to tf.data.Dataset for speed
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# create model
model = EoTNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate) # adam optimizer
loss_fn = tf.keras.losses.MeanSquaredError() # mse for loss

# for visualization
train_losses = []
val_losses = []
batch_count = 0  # global batch counter
batch_indices = []  # x-axis for batch-wise plot

# callbacks
best_val_loss = float('inf')
no_improve_count = 0
save_dir = "/content/drive/MyDrive/colab_checkpoints"
# save_dir = "src/reconstructionV2"
graph_dir = "src/reconstructionV2/graph_checkpoints"
patience = 15

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y_true=y, y_pred=preds)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(x, y):
    preds = model(x, training=False)
    return loss_fn(y_true=y, y_pred=preds)

for epoch in range(epochs):
    print(f"epoch {epoch+1}\n")

    # gradient descent
    train_loss = 0.0
    train_samples = 0
    pbar = tqdm(train_dataset, desc=f"training {epoch+1}/{epochs}", leave=False)
    for x, y in pbar:
        loss = float(train_step(x, y).numpy())
        train_losses.append(loss)
        batch_indices.append(batch_count)
        batch_count += 1
        batch_size_curr = int(x.shape[0])
        train_loss += loss * batch_size_curr
        train_samples += batch_size_curr
        pbar.set_postfix(loss=f"{train_loss / train_samples:.6f}")
    
    if train_samples == 0:
        print("skipping")
        continue # not enough samples
    
    train_loss /= train_samples # true sample-weighted average

    # evaluate
    val_loss = 0.0
    val_samples = 0
    val_pbar = tqdm(val_dataset, desc=f"validation {epoch+1}/{epochs}", leave=False)
    for x, y in val_pbar:
        loss = float(val_step(x, y).numpy())
        val_losses.append(loss)
        batch_size_curr = int(x.shape[0])
        val_loss += loss * batch_size_curr
        val_samples += batch_size_curr
        val_pbar.set_postfix(loss=f"{val_loss / val_samples:.6f}")
    
    if val_samples == 0:
        print("skipping")
        continue
    
    val_loss /= val_samples # true sample weighted average over epoch
    
    val_losses.append(val_loss)

    # model checkpointing + early stopping
    if val_loss <= best_val_loss:
        print(f"saving new best model with loss: {val_loss}")
        model.save(os.path.join(save_dir, "eot_best.keras")) # save model
        best_val_loss = val_loss # update loss
        no_improve_count = 0
    else:
        no_improve_count += 1
    
    if no_improve_count > patience:
        print("model stopping improving, ending training")
        break

    # save graphs   
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(batch_indices, train_losses, label="train loss", alpha=0.7)
    ax.plot(batch_indices[:len(val_losses)], val_losses, label="val loss", alpha=0.7)
    ax.set_xlabel("Batch #")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training vs Validation Loss (batch-wise)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, "loss_curve_batchwise.png"))
    plt.close(fig)