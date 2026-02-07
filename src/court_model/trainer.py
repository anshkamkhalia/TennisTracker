# trains the model on saved batches to detect court keypoints
# DEPRECATED 
import numpy as np
import tensorflow as tf
from model import CourtDetector
import glob
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# directories
dataset_dir = "data/court-model/data/datasets"

# load batch files
X_files = sorted(glob.glob(f"{dataset_dir}/X_batch_*.npy"))
y_files = sorted(glob.glob(f"{dataset_dir}/y_batch_*.npy"))

# create model
model = CourtDetector()

# optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()  # regression
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])

# model checkpoint - saves best model during training
model_checkpoint = ModelCheckpoint(
    'serialized_models/court_keypoint_detector.keras',         # file to save
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
    patience=10,             # how many epochs to wait before stopping
    restore_best_weights=True
)
# training loop
EPOCHS = 50
BATCH_SIZE = 4

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # shuffle batches every epoch
    perm = np.random.permutation(len(X_files))
    X_files = [X_files[i] for i in perm]
    y_files = [y_files[i] for i in perm]

    epoch_loss = []
    
    for X_file, y_file in zip(X_files, y_files):
        X_batch = np.load(X_file)
        y_batch = np.load(y_file)

        # optionally normalize keypoints to [0,1]
        X_batch = X_batch.astype('float32')
        y_batch = y_batch.astype('float32')
        img_h, img_w = X_batch.shape[1], X_batch.shape[2]
        y_batch = y_batch.reshape(y_batch.shape[0], -1)
        y_batch[:, ::2] /= img_w  # normalize x
        y_batch[:, 1::2] /= img_h  # normalize y

        # train on this batch
        history = model.fit(X_batch, y_batch, batch_size=BATCH_SIZE, verbose=1, validation_split=0.1, callbacks=[early_stopping, model_checkpoint, reduce_lr])
        batch_loss = history.history['loss'][0]
        epoch_loss.append(batch_loss)

    print(f"epoch {epoch+1} average loss: {np.mean(epoch_loss):.4f}")