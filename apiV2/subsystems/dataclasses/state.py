import numpy as np
import cv2 as cv

class PipelineState:

    def __init__(self):

        self.cap = None
        self.out = None

        self.frame_buffer = []
        self.frame_index = 0
        self.previous_prediction_sc = "neutral"
        self.last_pred_frame_sc = -999

        self.total_frames = None

        self.input_path = ""
        self.output_path = ""

        self.input_fps = 0
        self.dt = 0

        self.speed_buffer_size = 0

        # mini court setup
        self.court_box = None
        # for homography
        self.court_corners = None
        self.H = None
        self.k = 1 # for court shrinkage
        self.court_padding = 30 # amount of pixels to increase court size by

        # mini court
        self.mini_w, mini_h = 200, 400           # mini court size
        self.margin = 20                          # top-right corner padding
        self.mx, self.my = 1280 - self.mini_w - self.margin, self.margin
        self.mw, self.mh = self.mini_w, self.mini_h


        # create overlay with fully black opaque background
        self.mini_court_overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv.rectangle(self.mini_court_overlay, (self.mx, self.my), (self.mx + self.mw, self.my + self.mh), (0, 0, 0), -1)  # BLACK background

        # neon green lines
        self.line_color = (57, 255, 20)
        self.thick = 2

        # scale ratios based on real tennis court dimensions
        self.singles_ratio = 27 / 36
        self.net_y = self.my + self.mh // 2

        # bigger service boxes
        self.service_offset = int(self.mh * 0.25)  # ~1/4 from baseline to net

        # create court (alleys, boxes, lines, net, etc)
        cv.line(self.mini_court_overlay, (self.mx, self.my), (self.mx, self.my + self.mh), self.line_color, self.thick)           # left doubles
        cv.line(self.mini_court_overlay, (self.mx + self.mw, self.my), (self.mx + self.mw, self.my + self.mh), self.line_color, self.thick) # right doubles

        self.singles_offset = int(self.mw * (1 - self.singles_ratio) / 2)

        cv.line(self.mini_court_overlay, (self.mx + self.singles_offset, self.my), (self.mx + self.singles_offset, self.my + self.mh), self.line_color, self.thick)
        cv.line(self.mini_court_overlay, (self.mx + self.mw - self.singles_offset, self.my), (self.mx + self.mw - self.singles_offset, self.my + self.mh), self.line_color, self.thick)
        cv.line(self.mini_court_overlay, (self.mx, self.my), (self.mx + self.mw, self.my), self.line_color, self.thick)           # far baseline
        cv.line(self.mini_court_overlay, (self.mx, self.my + self.mh), (self.mx + self.mw, self.my + self.mh), self.line_color, self.thick) # near baseline
        cv.line(self.mini_court_overlay, (self.mx, self.net_y), (self.mx + self.mw, self.net_y), self.line_color, self.thick)

        cv.line(self.mini_court_overlay, (self.mx + self.singles_offset, self.my + self.service_offset),
                (self.mx + self.mw - self.singles_offset, self.my + self.service_offset), self.line_color, self.thick)
        cv.line(self.mini_court_overlay, (self.mx + self.singles_offset, self.my + self.mh - self.service_offset),
                (self.mx + self.mw - self.singles_offset, self.my + self.mh - self.service_offset), self.line_color, self.thick)

        center_x = self.mx + self.mw // 2
        cv.line(self.mini_court_overlay, (center_x, self.my + self.service_offset),
        (center_x, self.my + self.mh - self.service_offset), self.line_color, self.thick)

        self.audio_path = "api/temp_videos/audio.wav"
        self.video_path = None

        self.audio = None
        self.sr = 0
        self.time = None
        self.avg_energy = None
        self.MIN_ENERGY = None

        self.n_contacts = 0
        self.hit = False
        self.COOLDOWN_FRAMES = 0
        self.last_hit_frame = 0
