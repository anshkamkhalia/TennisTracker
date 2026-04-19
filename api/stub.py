from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import random
import numpy as np

app = FastAPI(title="app stub")

@app.post("/process-video")
async def main(request: Request, video: UploadFile = File(...)):
    
    return {
        "message": "saved successfully",
        "url": "public_url",
        "n_shots_by_poi": 10,
        "total_shots": 18,
        "forehand_percent": 70,
        "backhand_percent": 10,
        "slice_volley_percent": 10,
        "serve_overhead_percent": 10, 
        "right_wrist_avg": 10.6,
        "left_wrist_avg": 9.4,
        "right_wrist_v": [np.random.rand() + random.randint(5,25) for i in range(30)],
        "left_wrist_v": [np.random.rand() + random.randint(5,25) for i in range(30)],
        "heatmap": None,
        "ball_speeds": None,
    }