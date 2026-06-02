import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
from src.shot_classification.model import ShotClassifier
from src.shot_classification.neutral_model import NeutralIdentifier, Attention

class Models:

    def __init__(self):

        self.shot_model_path = "api/serialized_models/new_sc.keras" # use new shot clasification model (temporal transformer)
        self.neutral_model_path = "api/serialized_models/neutrality.keras"
        # classifies shots
        
        self.shot_classifier = tf.keras.saving.load_model(self.shot_model_path, custom_objects={ # load model with correct classes
            "ShotClassifier": ShotClassifier,
        })

        # identifies neutral positions
        self.neutral_identifier = tf.keras.saving.load_model(self.neutral_model_path, custom_objects={
            "NeutralIdentifier": NeutralIdentifier,
            "Attention": Attention,
        })

        # used for cropping player
        self.human_detector = YOLO("yolo11n.pt")

        # tracks balls across frames
        self.ball_tracker = YOLO("hugging_face_best.pt")

        # detects the bounding box of court
        self.court_detector = YOLO("src/court_model/best.pt")

        # mediapipe pose instance
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )