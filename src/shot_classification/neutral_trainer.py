# trains and exports a neutralilty classifier

from neutral_model import NeutralIdentifier, Attention
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os

LANDMARKS = "data/shot-classification/neutral_landmarks" # root for landmark data

# load data
X_train = np.load(
    os.path.join(
        LANDMARKS, "X_train_shot_classification_neutral.npy"
    )
)
X_test = np.load(
    os.path.join(
        LANDMARKS, "X_test_shot_classification_neutral.npy"
    )
)

y_train = np.load(
    os.path.join(
        LANDMARKS, "y_train_shot_classification_neutral.npy"
    )
)
y_test = np.load(
    os.path.join(
        LANDMARKS, "y_test_shot_classification_neutral.npy"
    )
)

# shuffle train
perm = np.random.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

# shuffle test
test_perm = np.random.permutation(len(X_test))
X_test = X_test[test_perm]
y_test = y_test[test_perm]

X_train = X_train[..., np.newaxis]  # shape becomes (num_samples, 33, 3, 1)
X_test  = X_test[..., np.newaxis]

# create model instance
neutrality = NeutralIdentifier()

# callbacks

# model checkpoint - saves best model during training
model_checkpoint = ModelCheckpoint(
    'serialized_models/neutrality.keras',         # file to save
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
    monitor='val_accuracy',      # what to monitor
    patience=15,             # how many epochs to wait before stopping
    restore_best_weights=True
)

# compile model
neutrality.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy", # multiple labels
    metrics=['accuracy']
)

# higher weightage
class_weights = {
    0: 5.0, 
    1: 1.0,
}

neutrality.fit(
    X_train,
    y_train,
    epochs=100,
    callbacks=[reduce_lr, model_checkpoint, early_stopping],
    batch_size=16,
    validation_data=(X_test, y_test)
)