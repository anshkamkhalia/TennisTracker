# config variables
previous_prediction = "neutral" # to save the last prediction
frame_index = 0 # to keep track of current frame
last_pred_frame = -999  # initialize far back so no text at start
state = None # will be 0 (neutral) or 1 (swinging)
output_class = -1 # forehand (0) or backhand (1) or others
trail = [] # stores ball trail
MAX_TRAIL_LENGTH = 40
coordinates = [] # stores ball locations
ball_history = [] # for savitzky–golay filter
BALL_SMOOTH_WINDOW = 5
BALL_POLY_ORDER = 2 # polynomial order for savitzky-golay, adjust as needed
view_type = None # either "top" or "court"
view_type_determined = False
pbar = None

# pixel to meters for ball speed
meters_per_pixel = None
court_baseline_length_meters = 23.77
court_box_updated = False
speed_buffer = []
speed_buffer_size = 50 # ~1 second (will reset after fps is known)
mps_to_mph_conversion_factor = 2.2369363 # conversion rate from mps to mph
prev_velocity = None
last_ball_px = None

# wrist velocity
r_wrist_buffer = []
l_wrist_buffer = []
WRIST_BUFFER_MAXLEN = 10
wrist_alpha = 0.5
r_vel_mps_display = 0.0
l_vel_mps_display = 0.0
r_vel_mph_display = 0.0
l_vel_mph_display = 0.0
PLAYER_HEIGHT_METERS = 1.82 # an assumption

# velocity graphs
velocities = []
l_w_velocities = []
r_w_velocities = []

# player movement
heat_color = None

# state variables
prev_court_preds = None
prev_frame_pose = None
prev_pose_results = None
prev_ball_frame = None
prev_ball_results = None
prev_player_movement_results = None
prev_player_movement_results_frame = None
prev_pose_landmarks = None
prev_pose_frame = None
pose_frame = None

# analytics
shot_occurences = {
    "forehand": 0,
    "backhand": 0,
    "serve_overhead": 0,
}
total_shots_for_percentages = 0