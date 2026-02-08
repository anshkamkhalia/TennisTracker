# uses on the fly preprocessing to train a model
# reduces OOM, blobs of .npy files, and batching issues
# uses lazy loading with a tf.data.Data object and generators

import tensorflow as tf
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from model import TrackNet
import torch
import os
import gc

tf.keras.mixed_precision.set_dtype_policy('mixed_float16')

teacher = YOLO("hugging_face_best.pt").to("cpu") # used for labeling; force cpu to avoid pytorch and tensorflow GPU conflicts
tracknet = TrackNet() # this is what we are training
video_dir = "src/ball_tracking/ball_tracking_data"
EPOCHS = 100

def video_frame_generator(video_dir):

    """uses a generator to load video frames"""

    cap = cv.VideoCapture(video_dir)
    while True:
        ret, frame = cap.read()
        if not ret: # break if final frame
            break
        frame = cv.resize(frame, (1280, 720)) # resize to a lower resolution

        # use teacher model to get candidates for labels
        with torch.no_grad():
            results = teacher.predict(
                source=frame,
                conf=0.25,
                stream=False,
                verbose=False,
                device="cpu",
            )[0]

            best_conf = 0
            best_box = None

            # get box with highest confidence
            for box in results.boxes:
                if box.conf[0] >= best_conf:
                    best_box = box
                    best_conf = box.conf[0]
            
            if best_box is not None:

                # create gaussian blur for TrackNet
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # get center (x,y)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                H, W = frame.shape[:2]

                y = np.arange(H)[:, None]
                x = np.arange(W)[None, :]

                heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 10**2))
                heatmap = heatmap[..., np.newaxis]
                heatmap = heatmap.astype(np.float16)  # 50% reduction
                yield frame, heatmap

            else:
                # create an empty heatmap
                H, W = frame.shape[:2]  # use actual frame shape
                heatmap = np.zeros((H, W, 1), dtype=np.float16)

                yield frame, heatmap

videos = sorted(os.listdir(video_dir)) # get videos

# split train and val videos
train_videos = ["videoplayback8.mp4", "videplayback5.mp4", "videoplayback4.mp4"]
val_videos = ["videoplayback2.mp4"]

loss_fn = tf.keras.losses.MeanSquaredError() # loss function
optimizer = tf.keras.optimizers.Adam(1e-4) # adam optimizer

# create tf.data.Dataset object for testing
test_dataset = (

    tf.data.Dataset.from_generator(

        lambda: video_frame_generator(os.path.join(video_dir, val_videos[0])), # generator function
        output_signature=(
            tf.TensorSpec(shape=(720, 1280, 3), dtype=tf.float16),  # inputs
            tf.TensorSpec(shape=(720, 1280, 1), dtype=tf.float16) # outputs
        )

    )
    .batch(2) # batch size 2
    .prefetch(tf.data.AUTOTUNE) # prefetch for maximum speed

)

best_test_loss = float("inf") # for model checkpointing

for video in train_videos:

    # create tf.data.Dataset object
    dataset = (

        tf.data.Dataset.from_generator(

            lambda: video_frame_generator(os.path.join(video_dir, video)), # generator function
            output_signature=(
                tf.TensorSpec(shape=(720, 1280, 3), dtype=tf.float16),  # inputs
                tf.TensorSpec(shape=(720, 1280, 1), dtype=tf.float16) # outputs
            )

        )
        .shuffle(50) # shuffle inputs
        .batch(2) # batch size 2
        .prefetch(tf.data.AUTOTUNE) # prefetch for maximum speed

    )

    # custom training loop
    for epoch in range(EPOCHS):
        print(f"epoch {epoch+1}\n")

        train_loss = 0.0 # initialize at the beginning of each epoch
        train_samples = 0
        for x, y in dataset:
            with tf.GradientTape() as tape: # the glorious gradient tape
                preds = tracknet(x, training=True) # get predictions
                loss = loss_fn(y, preds) # get loss

            grads = tape.gradient(loss, tracknet.trainable_variables) # find gradients
            optimizer.apply_gradients(zip(grads, tracknet.trainable_variables)) # apply gradients

            train_loss += loss.numpy() # increment train loss
            train_samples += x.shape[0] # 2 samples per batch

            # aggressive garbage collection
            del x, y, grads
            gc.collect()

        train_loss /= train_samples # average

        test_loss = 0.0 # initialize
        test_samples = 0 # for averaging
        for x,y in test_dataset:
            preds = tracknet(x, training=False) # get predictions
            loss = loss_fn(y, preds) # calculate error
            test_loss += loss.numpy() # increment
            test_samples += x.shape[0] # 2 samples per batch

            # aggressive garbage collection
            del x, y, preds
            gc.collect()

        test_loss /= test_samples # average over dataset

        print(f"train_loss: {train_loss} | test_loss {test_loss}") # logging

        # model checkpointing
        if test_loss < best_test_loss: 
            best_test_loss = test_loss
            tracknet.save("serialized_models/tracknet_best.keras") # save best model so far
            print(f"saved to serialized_models/tracknet_best.keras, test_loss: {test_loss}")