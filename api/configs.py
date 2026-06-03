import os
from dotenv import load_dotenv # secure credential storage

class PipelineState:

    def __init__(self):

        # config variables
        self.previous_prediction = "neutral" # to save the last prediction
        self.frame_index = 0 # to keep track of current frame
        self.last_pred_frame = -999  # initialize far back so no text at start
        self.state = None # will be 0 (neutral) or 1 (swinging)
        self.output_class = -1 # forehand (0) or backhand (1) or others
        self.trail = [] # stores ball trail
        self.MAX_TRAIL_LENGTH = 40
        self.coordinates = [] # stores ball locations
        self.ball_history = [] # for savitzky–golay filter
        self.BALL_SMOOTH_WINDOW = 5
        self.BALL_POLY_ORDER = 2 # polynomial order for savitzky-golay, adjust as needed
        self.view_type = None # either "top" or "court"
        self.view_type_determined = False
        self.pbar = None

        # pixel to meters for ball speed
        self.meters_per_pixel = None
        self.court_baseline_length_meters = 23.77
        self.court_box_updated = False
        self.speed_buffer = []
        self.speed_buffer_size = 50 # ~1 second (will reset after fps is known)
        self.mps_to_mph_conversion_factor = 2.2369363 # conversion rate from mps to mph
        self.prev_velocity = None
        self.last_ball_px = None

        # wrist velocity
        self.r_wrist_buffer = []
        self.l_wrist_buffer = []
        self.WRIST_BUFFER_MAXLEN = 10
        self.wrist_alpha = 0.5
        self.r_vel_mps_display = 0.0
        self.l_vel_mps_display = 0.0
        self.r_vel_mph_display = 0.0
        self.l_vel_mph_display = 0.0
        self.PLAYER_HEIGHT_METERS = 1.82 # an assumption

        # velocity graphs
        self.velocities = []
        self.l_w_velocities = []
        self.r_w_velocities = []

        # player movement
        self.heat_color = None

        # state variables
        self.prev_court_preds = None
        self.prev_frame_pose = None
        self.prev_pose_results = None
        self.prev_ball_frame = None
        self.prev_ball_results = None
        self.prev_player_movement_results = None
        self.prev_player_movement_results_frame = None
        self.prev_pose_landmarks = None
        self.prev_pose_frame = None
        self.pose_frame = None

        # analytics
        self.shot_occurences = {
            "forehand": 0,
            "backhand": 0,
            "serve_overhead": 0,
        }
        self.total_shots_for_percentages = 0

        self.pose_landmarks_3d = []

# load environment variables for cloudflare r2 connnection
load_dotenv()

# get r2 credentials
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_KEY = os.getenv("R2_KEY")
R2_SECRET = os.getenv("R2_SECRET")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL")

MAX_VIDEO_SIZE = 150 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",   # .mov
    "video/x-matroska"   # .mkv
}
ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv"}