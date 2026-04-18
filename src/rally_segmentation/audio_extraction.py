import subprocess
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

video_path = "api/videoplayback10.mp4"
audio_path = "src/rally_segmentation/audio.wav"

graph = sys.argv[1]

# extract audio from video using ffmpeg
if not os._exists(audio_path):
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vn",              
        "-acodec", "pcm_s16le",
        "-ar", "16000",     
        "-ac", "1", "-y",
        audio_path
    ], check=True)

audio, sr = librosa.load(audio_path, sr=16_000) # load .wav audio with librosa
time = np.arange(len(audio)) / sr

if graph == "w":

    # waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.title("audio waveform")
    plt.show()

else:

    # spectrogram
    D = librosa.amplitude_to_db(
        abs(librosa.stft(audio)),
        ref=np.max
    )

    plt.figure(figsize=(12, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.show()