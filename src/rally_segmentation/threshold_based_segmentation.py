import subprocess
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import time as t

video_path = "api/videoplayback6.mp4"
audio_path = "src/rally_segmentation/audio.wav"

# extract audio from video using ffmpeg
if not os.path.exists(audio_path):
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vn",              
        "-acodec", "pcm_s16le",
        "-ar", "16000",     
        "-ac", "1", "-y",
        audio_path
    ], check=True)

# SUPER IMPORTANT THRESHOLDS
WAIT_TIME = 90 # amount of frames (30 fps)

audio, sr = librosa.load(audio_path, sr=16_000) # load .wav audio with librosa
time = np.arange(len(audio)) / sr

def get_audio_at_frame(frame_idx):
    time = frame_idx / fps
    sample_idx = int(time * sr)

    # small window (e.g. 50 ms)
    window = int(0.05 * sr)

    start = max(0, sample_idx - window // 2)
    end = min(len(audio), sample_idx + window // 2)

    return audio[start:end]

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
frame_index = 0

n_contacts = 0
hit = False
delay = 1 / 30

MIN_ENERGY = 0.05
COOLDOWN_FRAMES = int(0.3 * fps)
last_hit_frame = -COOLDOWN_FRAMES

# How long to show "CONTACT DETECTED" after a hit (in frames)
CONTACT_DISPLAY_FRAMES = int(1.0 * fps)  # 1 second
last_display_hit_frame = -CONTACT_DISPLAY_FRAMES

while True:
    start = t.time()
    ret, frame = cap.read()
    if not ret:
        break

    current_audio = get_audio_at_frame(frame_index)
    energy = np.sqrt(np.mean(current_audio ** 2))

    hit = False

    if energy >= MIN_ENERGY and frame_index - last_hit_frame > COOLDOWN_FRAMES:
        hit = True
        n_contacts += 1
        last_hit_frame = frame_index
        last_display_hit_frame = frame_index

    # Show "CONTACT DETECTED" for 1 second after a hit
    show_contact = (frame_index - last_display_hit_frame) < CONTACT_DISPLAY_FRAMES

    if show_contact:
        label = "CONTACT DETECTED"
        text_color = (0, 255, 0)       # bright green text
        bg_color = (0, 60, 0)          # dark green background

        # Measure text size for background rectangle
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        (text_w, text_h), baseline = cv.getTextSize(label, font, font_scale, thickness)

        pad = 12
        x, y = 50, 100
        # Draw filled background rectangle
        cv.rectangle(frame,
                     (x - pad, y - text_h - pad),
                     (x + text_w + pad, y + baseline + pad),
                     bg_color, -1)
        # Draw a bright green border
        cv.rectangle(frame,
                     (x - pad, y - text_h - pad),
                     (x + text_w + pad, y + baseline + pad),
                     text_color, 2)
        # Draw text
        cv.putText(frame, label, (x, y), font, font_scale, text_color, thickness)
    else:
        label = "waiting"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        text_color = (160, 160, 160)   # gray text
        bg_color = (40, 40, 40)        # dark gray background

        (text_w, text_h), baseline = cv.getTextSize(label, font, font_scale, thickness)

        pad = 12
        x, y = 50, 100
        cv.rectangle(frame,
                     (x - pad, y - text_h - pad),
                     (x + text_w + pad, y + baseline + pad),
                     bg_color, -1)
        cv.rectangle(frame,
                     (x - pad, y - text_h - pad),
                     (x + text_w + pad, y + baseline + pad),
                     text_color, 2)
        cv.putText(frame, label, (x, y), font, font_scale, text_color, thickness)

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = t.time() - start
    t.sleep(max(0, delay - elapsed))

    frame_index += 1

print(n_contacts)