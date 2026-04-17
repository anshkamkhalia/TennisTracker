import tensorflow as tf
from model import ContactTransformer
import numpy as np

# configs
EPOCHS = 100
BATCH_SIZE = 16

train_file_idxs = ["1", "2", "3"]
val_file_idxs = ["4"]

X_list_tr = []
y_list_tr = []

X_list_val = []
y_list_val = []

def load_file(idx):
    X = np.load(f"src/rally_segmentation/processed_data/{idx}_X.npy")
    y = np.load(f"src/rally_segmentation/processed_data/{idx}_y.npy")
    
    if idx in train_file_idxs:
        X_list_tr.append(X)
        y_list_tr.append(y)
    else:
        X_list_val.append(X)
        y_list_val.append(y)

for idx in train_file_idxs:
    load_file(idx)
for idx in val_file_idxs:
    load_file(idx)

X_train, y_train = np.concatenate(X_list_tr, axis=0), np.concatenate(y_list_tr, axis=0)
X_test, y_test = np.concatenate(X_list_val, axis=0), np.concatenate(y_list_val, axis=0)

# model setup
model = ContactTransformer()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="serialized_models/contact_best.keras",
    monitor="val_loss",
    save_best_only=True
)

optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
    optimizer=optimizer,
)

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_test, y_test)
)