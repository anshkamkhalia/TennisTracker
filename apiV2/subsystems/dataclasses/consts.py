import boto3
import os
from dotenv import load_dotenv
from botocore.client import Config

load_dotenv()

class PipelineConstants:

    def __init__(self):
        
        # cloudflare credentials
        self.R2_ENDPOINT = os.getenv("R2_ENDPOINT")
        self.R2_KEY = os.getenv("R2_KEY")
        self.R2_SECRET = os.getenv("R2_SECRET")
        self.R2_BUCKET = os.getenv("R2_BUCKET")
        self.R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

        self.MAX_VIDEO_SIZE = 150 * 1024 * 1024
        self.ALLOWED_MIME_TYPES = {
            "video/mp4",
            "video/quicktime",   # .mov
            "video/x-matroska"   # .mkv
        }
        self.ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv"}

        # bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.R2_ENDPOINT,
            aws_access_key_id=self.R2_KEY,
            aws_secret_access_key=self.R2_SECRET,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

        # shot classification
        self.SHOT_LABELS = {
            "forehand": 0,
            "backhand": 1,
            "serve_overhead": 2,
        }

        self.SHOT_LABELS_INV = {v:k for k, v in self.SHOT_LABELS.items()} # invert