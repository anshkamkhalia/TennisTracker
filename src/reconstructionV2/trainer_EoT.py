# trains EoT model

import tensorflow as tf
from eot_model import EoTNetwork
import numpy as np

# hyperparams
seq_len = 60
batch_size = 64
epochs = 50
learning_rate = 1e-3
data_dir = "src/reconstructionV2/synthetic_data"
strokes = ["groundstroke", "serve", "lob"]
num_batches_per_type = 30
validation_split = 0.1

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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae'])

# callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('serialized_models/eot_model_best.keras', monitor='val_loss', save_best_only=True)
]

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks,
)