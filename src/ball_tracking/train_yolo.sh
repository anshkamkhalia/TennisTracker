#!/bin/bash

# yolo training command
echo "starting yolo training"

yolo task=detect mode=train \
  model=yolo11n.pt \
  data=data/ball-tracking/tennis-ball-detection-yolov11/data.yaml \
  epochs=50 \
  batch=8 \
  imgsz=960 \
  lr0=0.001 \
  augment=True \
  mosaic=1.0 \
  mixup=0.2 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  degrees=10 \
  translate=0.1 \
  scale=0.5 \
  fliplr=0.5 \
  plots=True \
  > src/ball_tracking/training_log.txt 2>&1

echo "completed training. saved to training_log.txt"
