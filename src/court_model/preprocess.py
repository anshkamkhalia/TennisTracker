# exports data as npy files for court keypoint detection
# DEPRECATED

import cv2 as cv
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# directories
ROOT = "data/court-model"
images_dir = os.path.join(ROOT, "data/images")
train_json_path = f"{ROOT}/data/data_train.json"
test_json_path = f"{ROOT}/data/data_val.json"
output_dir = os.path.join(ROOT, "datasets")
os.makedirs(output_dir, exist_ok=True)

# load json
with open(train_json_path, "r") as f:
    train_json = json.load(f)

with open(test_json_path, "r") as f:
    test_json = json.load(f)

# load json of ids and their corresponding keypoints
id_and_kp = {}
for dictionary in train_json + test_json:
    current_id = dictionary["id"]
    id_and_kp[current_id] = dictionary["kps"]

# list contents
filenames = os.listdir(images_dir)

# batch params
BATCH_SIZE = 100
x_batch, y_batch = [], []
batch_counter = 0

# main processing loop
for idx, filename in enumerate(filenames):
    img_path = os.path.join(images_dir, filename)
    img = cv.imread(img_path)  # read image data
    if img is None:
        print(f"Warning: {filename} could not be read, skipping")
        continue

    img = img.astype('float32') / 255.0  # normalize

    img_id = filename.replace(".png", "").strip()

    if img_id not in id_and_kp:
        print(f"keypoints not found for {img_id}, skipping")
        continue

    x_batch.append(img)
    y_batch.append(id_and_kp[img_id])

    # save batch when full
    if len(x_batch) == BATCH_SIZE or idx == len(filenames) - 1:
        batch_counter += 1
        np.save(os.path.join(output_dir, f"X_batch_{batch_counter}.npy"), np.array(x_batch, dtype='float32'))
        np.save(os.path.join(output_dir, f"y_batch_{batch_counter}.npy"), np.array(y_batch, dtype='float32'))
        print(f"{batch_counter} saved with {len(x_batch)} samples")

        # clear for next batch
        x_batch, y_batch = [], []

print("batches saved")