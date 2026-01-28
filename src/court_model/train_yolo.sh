#!/bin/bash

echo "starting yolo training"

yolo task=detect mode=train \
  model=yolo11n.pt \
  data=yolov11-tennis-court-data \
  epochs=80 \
  batch=8 \
  imgsz=960 \
  lr0=0.001 \
  augment=True \
  mosaic=0.2 \
  mixup=0.0 \
  hsv_h=0.005 \
  hsv_s=0.3 \
  hsv_v=0.3 \
  degrees=2 \
  translate=0.02 \
  scale=0.2 \
  fliplr=0.0 \
  plots=True \
  > src/shot_scoring/training_log.txt 2>&1

echo "completed training. saved to training_log.txt"
