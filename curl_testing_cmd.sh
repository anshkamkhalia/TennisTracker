#!/bin/bash

curl -v -X POST http://localhost:8000/process-video \
    -F "video=@/Users/ansh/Downloads/development/tennistracker/api/test_videos/videoplayback8.mp4;type=video/mp4" \
    -o response.json