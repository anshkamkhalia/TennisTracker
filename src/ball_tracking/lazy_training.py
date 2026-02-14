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

tf.keras.mixed_precision.set_dtype_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

teacher = YOLO("hugging_face_best.pt").to("cpu") # used for labeling; force cpu to avoid pytorch and tensorflow GPU conflicts
tracknet = TrackNet() # this is what we are training
dummy_input = tf.zeros((1, 360, 640, 3), dtype=tf.float32)
_ = tracknet(dummy_input)  # now model builds
video_dir = "src/ball_tracking/ball_tracking_data"
EPOCHS = 10
BATCH_SIZE = 2

def make_gaussian_kernel(size=41, sigma=5):
    """builds a small gaussian kernel to avoid creating multiple full heatmaps"""

    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel.astype(np.float16)

GAUSSIAN_KERNEL = make_gaussian_kernel()
K = GAUSSIAN_KERNEL.shape[0] // 2
EMPTY_HEATMAP = np.zeros((360, 640, 1), dtype=np.float16)

@tf.function
@tf.keras.saving.register_keras_serializable(package="custom_loss")
def tracknet_loss(y_true, y_pred, eps=0.001, lambda_bg=0.005):
    """
    tracknet-style foreground / background loss
    y_true, y_pred: (B, H, W, 1)
    """

    # force float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # clip loss
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    # foreground mask (ball region)
    fg_mask = tf.cast(y_true > eps, tf.float32)

    # background mask
    bg_mask = 1.0 - fg_mask

    # foreground loss (gaussian regression)
    fg_loss = tf.reduce_sum(
        fg_mask * tf.square(y_true - y_pred)
    ) / (tf.reduce_sum(fg_mask) + 1e-6)

    # background loss (suppress noise)
    bg_loss = tf.reduce_sum(
        bg_mask * tf.square(y_pred)
    ) / (tf.reduce_sum(bg_mask) + 1e-6)

    # combined loss
    total_loss = fg_loss # + lambda_bg * bg_loss
    return total_loss

def video_frame_generator(video_dir):

    """uses a generator to load video frames"""

    cap = cv.VideoCapture(video_dir)
    while True:
        ret, frame = cap.read()
        if not ret: # break if final frame
            break
        frame = cv.resize(frame, (640, 360))

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
                
                # get center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                H, W = 360, 640
                heatmap = np.zeros((H, W), dtype=np.float32)

                # bounds for stamping
                x0 = max(0, cx - K)
                x1 = min(W, cx + K + 1)
                y0 = max(0, cy - K)
                y1 = min(H, cy + K + 1)

                # kernel slice bounds
                kx0 = K - (cx - x0)
                kx1 = K + (x1 - cx)
                ky0 = K - (cy - y0)
                ky1 = K + (y1 - cy)

                # stamp
                heatmap[y0:y1, x0:x1] = GAUSSIAN_KERNEL[ky0:ky1, kx0:kx1]

                heatmap = heatmap[..., None]  # add channel dim
                frame = frame.astype(np.float16) / 255.0
                yield frame, heatmap

            else:
                # create an empty heatmap
                frame = frame.astype(np.float16) / 255.0
                yield frame, EMPTY_HEATMAP

    cap.release()

videos = sorted(os.listdir(video_dir)) # get videos

# split train and val videos
train_videos = ["videoplayback8.mp4", "videoplayback5.mp4", "videoplayback4.mp4"]
val_videos = ["videoplayback2.mp4"]

# loss_fn = tf.keras.losses.BinaryCrossentropy(dtype=tf.float32) # force float32
loss_fn = tracknet_loss
optimizer = tf.keras.optimizers.Adam(3e-4)

patience = 7

@tf.function(reduce_retracing=True)
def train_step(x, y):

    """optimizes training steps with @tf.function"""

    vars = tracknet.trainable_variables

    with tf.GradientTape() as tape:
        preds = tracknet(x, training=True)
        preds = tf.cast(preds, tf.float32) # cast to float32 to avoid destroying gradients
        # masked loss
        y = tf.cast(y, tf.float32)
        loss = loss_fn(y, preds)

    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss

best_global_test_loss = float("inf") # for model checkpointing
best_test_loss = float("inf") # also for model checkpointing
global_no_improve_count = 0 # for early stopping

for video in train_videos:
    no_improve_count = 0  # per-video early stop counter

    print(video)

    # create tf.data.Dataset object
    dataset = (

        tf.data.Dataset.from_generator(

            lambda: video_frame_generator(os.path.join(video_dir, video)), # generator function
            output_signature=(
                tf.TensorSpec(shape=(360, 640, 3), dtype=tf.float16),  # inputs
                tf.TensorSpec(shape=(360, 640, 1), dtype=tf.float16) # outputs
            )

        )
        .batch(BATCH_SIZE) # batch size n

    )
    print("train dataset defined")

    # create tf.data.Dataset object for testing
    test_dataset = (

        tf.data.Dataset.from_generator(

            lambda: video_frame_generator(os.path.join(video_dir, val_videos[0])), # generator function
            output_signature=(
                tf.TensorSpec(shape=(360, 640, 3), dtype=tf.float16),  # inputs
                tf.TensorSpec(shape=(360, 640, 1), dtype=tf.float16) # outputs
            )

        )
        .batch(BATCH_SIZE) # batch size n

    )
    print("val dataset defined")

    # custom training loop
    for epoch in range(EPOCHS):
        print(f"epoch {epoch+1}\n")

        train_loss = 0.0 # initialize at the beginning of each epoch
        train_samples = 0

        train_iter = iter(dataset) # avoid generator exhaustion

        for x,y in train_iter:

            loss = train_step(x,y)

            train_loss += loss.numpy() # increment train loss
            train_samples += x.shape[0] # 2 samples per batch

            # aggressive garbage collection
            del x, y

        if train_samples > 0:
            train_loss /= train_samples
        else:
            print("no train samples in this epoch")
            continue

        test_loss = 0.0 # initialize
        test_samples = 0 # for averaging

        test_iter = iter(test_dataset)

        for x,y in test_iter:
            preds = tracknet(x, training=False) # get predictions
            # use masked loss
            loss = loss_fn(y, preds)
            test_loss += loss.numpy() # increment
            test_samples += x.shape[0] # 2 samples per batch

            # to store val predictions
            if not os.path.exists("tracknet_results"):
                os.makedirs("tracknet_results")
            else: pass

            if test_samples % 200 == 0:
                os.makedirs(f"tracknet_results/epoch{epoch+1}", exist_ok=True)
                preds_to_save = np.array(preds[0, ..., 0] * 255).astype(np.uint8)  # remove batch and channel dim
                true_to_save = np.array(y[0, ..., 0] * 255).astype(np.uint8)
                # apply a colormap for visualization
                pred_colored = cv.applyColorMap(preds_to_save, cv.COLORMAP_JET)
                true_colored = cv.applyColorMap(true_to_save, cv.COLORMAP_JET)

                cv.imwrite(os.path.join(f"tracknet_results/epoch{epoch+1}", f"test_sample_pred{test_samples}.jpeg"), pred_colored)
                cv.imwrite(os.path.join(f"tracknet_results/epoch{epoch+1}", f"test_sample_true{test_samples}.jpeg"), true_colored)
                del true_to_save, preds_to_save

            # aggressive garbage collection
            del x, y, preds

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
            tracknet.save("serialized_models/tracknet_best.keras")  # global best
            print(f"saved GLOBAL best model, test_loss: {test_loss}")
        else:
            no_improve_count += 1
            global_no_improve_count += 1

        if no_improve_count > patience:
          break