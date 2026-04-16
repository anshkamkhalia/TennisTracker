import tensorflow as tf
from model import ContactTransformer
import numpy as np
from tqdm import tqdm
import gc

# configs
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
SHUFFLE = 300
PATIENCE = 10

train_file_idxs = ["1", "2", "3"]
val_file_idxs = ["4"]

# data generator
def datagen(idx):

    X = np.load(f"src/rally_segmentation/processed_data/{idx}_X.npy")
    y = np.load(f"src/rally_segmentation/processed_data/{idx}_y.npy")

    n_samples = X.shape[0]
    
    for start in range(0, n_samples, BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X[start:end].copy()
        y_batch = y[start:end].copy()

        yield X_batch, y_batch

# create datasets
training_dataset = tf.data.Dataset.from_generator(
    lambda: datagen(train_file_idxs),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, 15, 10)),
        tf.TensorSpec(shape=(BATCH_SIZE)),
    )
    .shuffle(SHUFFLE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: datagen(val_file_idxs),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, 15, 10)),
        tf.TensorSpec(shape=(BATCH_SIZE)),
    )
    .shuffle(SHUFFLE)
    .prefetch(tf.data.AUTOTUNE)
)

print("datasets created")

# loss, optimizers, and model
model = ContactTransformer()
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(LR)

@tf.function
def step(x, y, training):

    with tf.GradientTape() as tape:
        preds = model(x, training=training)
        y = tf.cast(y, tf.float32)
        preds = tf.cast(preds, tf.float32)

        loss = loss_fn(preds, y)

    if training:
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, vars))

    return loss

best_test_loss = float("inf") # for model checkpointing
no_improve_count = 0

idx_list = ["1", "2", "3", "4"]

def count_samples(idx_list):
    total = 0
    for idx in idx_list:
        arr = np.load(f"src/rally_segmentation/processed_data/{idx}_X.npy")
        total += arr.shape[0]
        del arr
    return total

steps_per_epoch = count_samples(idx_list=idx_list) // BATCH_SIZE

for epoch in range(EPOCHS):
    print(f"epoch {epoch+1}\n")

    train_loss = 0.0
    train_samples = 0 
    train_iter = iter(training_dataset) # avoid generator exhaustion

    # train model
    with tqdm(total=steps_per_epoch, desc=f"TRAIN: epoch {epoch+1}") as pbar:
        for x,y in train_iter:

            loss = step(x,y, training=True)
            train_loss += loss.numpy() * x.shape[0]
            train_samples += x.shape[0] # n samples per batch

            del x,y # instant gc

            avg_loss = train_loss / train_samples
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
            pbar.update(1)

        if train_samples > 0 :
            train_loss /= train_samples # average
        else:
            print("no train samples this epoch")
            continue

    # evaluate model
    test_loss = 0.0
    test_samples = 0
    test_iter = iter(val_dataset)

    with tqdm(total=steps_per_epoch, desc=f"VAL: epoch {epoch+1}") as pbar:
        for x,y in test_iter:

            loss = step(x,y, training=False)
            test_loss += loss.numpy() * x.shape[0]
            test_samples += x.shape[0] # n samples per batch

            del x,y # instant gc

            avg_loss = train_loss / train_samples
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})
            pbar.update(1)

        if test_samples > 0 :
            test_loss /= test_samples # average
        else:
            print("no test samples this epoch")
            continue

    print(f"train_loss: {train_loss} | test_loss {test_loss}") # logging

    # model checkpointing
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        no_improve_count = 0
        model.save("serialized_models/contact_detector_new.keras")  # global best
        print(f"saved best model, test_loss: {test_loss}")
    else:
        no_improve_count += 1

    if no_improve_count > PATIENCE:
        break

    gc.collect()