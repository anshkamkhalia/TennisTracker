# trains and exports a shot classification model

from model import ShotClassifier, Attention, SequenceAttention
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import sys

LANDMARKS = "data/shot-classification/landmarks" # root for landmark data

# the following code is very stupid
# im just too lazy to change it

# load saved data -> forehands
X_train_tf = np.load(os.path.join(
    LANDMARKS, "X_train_shot_classification_forehand.npy"
))
y_train_tf = np.load(os.path.join(
    LANDMARKS, "y_train_shot_classification_forehand.npy"
))

X_test_tf = np.load(os.path.join(
    LANDMARKS, "X_test_shot_classification_forehand.npy"
))
y_test_tf = np.load(os.path.join(
    LANDMARKS, "y_test_shot_classification_forehand.npy"
))

# load saved data -> backhands
X_train_tb = np.load(os.path.join(
    LANDMARKS, "X_train_shot_classification_backhand.npy"
))
y_train_tb = np.load(os.path.join(
    LANDMARKS, "y_train_shot_classification_backhand.npy"
))

X_test_tb = np.load(os.path.join(
    LANDMARKS, "X_test_shot_classification_backhand.npy"
))
y_test_tb = np.load(os.path.join(
    LANDMARKS, "y_test_shot_classification_backhand.npy"
))

# load saved data -> slices + volleys
X_train_sv = np.load(os.path.join(
    LANDMARKS, "X_train_shot_classification_slice_volley.npy"
))
y_train_sv = np.load(os.path.join(
    LANDMARKS, "y_train_shot_classification_slice_volley.npy"
))

X_test_sv = np.load(os.path.join(
    LANDMARKS, "X_test_shot_classification_slice_volley.npy"
))
y_test_sv = np.load(os.path.join(
    LANDMARKS, "y_test_shot_classification_slice_volley.npy"
))

# load saved data -> serves + overheads
X_train_so = np.load(os.path.join(
    LANDMARKS, "X_train_shot_classification_serve_overhead.npy"
))
y_train_so = np.load(os.path.join(
    LANDMARKS, "y_train_shot_classification_serve_overhead.npy"
))

X_test_so = np.load(os.path.join(
    LANDMARKS, "X_test_shot_classification_serve_overhead.npy"
))
y_test_so = np.load(os.path.join(
    LANDMARKS, "y_test_shot_classification_serve_overhead.npy"
))

# combine datasets
X_train = np.concatenate([
    X_train_tf,
    X_train_tb,
    X_train_sv,
    X_train_so
], axis=0)

y_train = np.concatenate([
    y_train_tf,
    y_train_tb,
    y_train_sv,
    y_train_so
], axis=0)

X_test = np.concatenate([
    X_test_tf,
    X_test_tb,
    X_test_sv,
    X_test_so
], axis=0)

y_test = np.concatenate([
    y_test_tf,
    y_test_tb,
    y_test_sv,
    y_test_so
], axis=0)

# shuffle train
perm = np.random.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

# shuffle test
test_perm = np.random.permutation(len(X_test))
X_test = X_test[test_perm]
y_test = y_test[test_perm]

# create model instance
shot_classifier = ShotClassifier()

# callbacks

# model checkpoint - saves best model during training
model_checkpoint = ModelCheckpoint(
    'serialized_models/shot_classifier_with_gelu.keras',  # file to save weights
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
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
    patience=7,             # how many epochs to wait before stopping
    restore_best_weights=True
)

# compile model
shot_classifier.compile(
    optimizer=Adam(0.001),
    loss="sparse_categorical_crossentropy", # multiple labels
    metrics=['accuracy']
)

# class_weight = {
#     0: 1.3,   # forehand 
#     1: 1.3,   # backhand
#     2: 1.15    # slice 
# }

# fit model
# shot_classifier.fit(
#     X_train,
#     y_train,
#     epochs=50,
#     callbacks=[reduce_lr, model_checkpoint, early_stopping],
#     batch_size=16,
#     validation_data=(X_test, y_test),
#     class_weight=class_weight
# )

# fit model
shot_classifier.fit(
    X_train,
    y_train,
    epochs=75,
    callbacks=[reduce_lr, model_checkpoint, early_stopping],
    batch_size=64,
    validation_data=(X_test, y_test),
)