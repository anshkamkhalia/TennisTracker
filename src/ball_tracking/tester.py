# tests the fine-tuned yolo model on actual videos

from ultralytics import YOLO

i = 5 # video" index to predict on
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instance
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on a video
results = model.track(
    source=video_path,   # input video
    conf=0.20,           # confidence threshold
    save=True,           # save output
    project="outputs",   # change output directory
)

# results contains per-frame predictions