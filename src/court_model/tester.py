# tests the court modeling

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import sys
from line_intersection import line_intersection

i = sys.argv[2]  # video index
video_path = f"data/court-level-videos/videoplayback{i}.mp4"

# load trained yolo instance
model = YOLO("hugging_face_best.pt")

# load video
cap = cv.VideoCapture(video_path)

# output video writer
out = cv.VideoWriter(
    f"outputs/run_court_model{i}.mp4",
    cv.VideoWriter_fourcc(*"mp4v"),
    30,
    (1280, 720)
)

videos60fps = [5,6,7,8] # videos that are more than 60fps
verbose = False if int(sys.argv[1]) == 0 else True # verbose flag
coordinates = [] # stores ball locations

# trail storage
trail = []
MAX_TRAIL_LENGTH = 50

frame_index = 0
keypoints = None

# clicked_points = []

# def get_points(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#         print(f"point registered: {(x, y)}")

# ret, frame = cap.read()
# frame = cv.resize(frame, (1280, 720))
# smoothing_trail = [] # for trajectory smoothing

# cv.imshow("click points", frame)             # create window first
# cv.setMouseCallback("click points", get_points)  # then register callback

# while True:
#     display = frame.copy()
#     for pt in clicked_points:
#         cv.circle(display, pt, 5, (0, 0, 255), -1)
#     cv.imshow("click points", display)

#     key = cv.waitKey(1) & 0xFF
#     if key == 27 or len(clicked_points) >= 4:  # ESC or 4 points
#         break

# cv.destroyAllWindows()
# cap.release()

# mini-court size (HxW)
# court_height = 300
# court_width = 150
# padding = 40

# # create the mini-court canvas
# court_canvas = np.zeros((court_height, court_width, 3), dtype=np.uint8)

# # draw court lines (green)
# cv.rectangle(
#     court_canvas,
#     (0, 0),
#     (court_width - 1, court_height - 1),
#     (0, 255, 0),
#     2
# )

# net_y = court_height // 2
# margin = court_width // 5
# service_box_length = int((21 / 39) * (court_height // 2))

# # net
# cv.line(court_canvas, (0, net_y), (court_width, net_y), (0, 255, 0), 2)

# # singles lines
# cv.line(court_canvas, (margin, 0), (margin, court_height), (0, 255, 0), 2)
# cv.line(court_canvas, (court_width - margin, 0),
#         (court_width - margin, court_height), (0, 255, 0), 2)

# # service box lines
# cv.line(
#     court_canvas,
#     (margin, net_y - service_box_length),
#     (court_width - margin, net_y - service_box_length),
#     (0, 255, 0),
#     2
# )
# cv.line(
#     court_canvas,
#     (margin, net_y + service_box_length),
#     (court_width - margin, net_y + service_box_length),
#     (0, 255, 0),
#     2
# )

# # center service line (TEE)
# cv.line(
#     court_canvas,
#     (court_width // 2, net_y - service_box_length),
#     (court_width // 2, net_y + service_box_length),
#     (0, 255, 0),
#     2
# )

# padded_court_height = court_height + 2 * padding
# padded_court_width = court_width + 2 * padding

# padded_court = np.zeros(
#     (padded_court_height, padded_court_width, 3),
#     dtype=np.uint8
# )

# padded_court[
#     padding:padding + court_height,
#     padding:padding + court_width
# ] = court_canvas

# load video
cap = cv.VideoCapture(video_path)
cy_prev1 = None
cy_prev2 = None
vy_prev = None
bounce = False

# bounce detection configs
ball_history = []        # stores (cx, cy)
bounce_points = []       # projected minimap points
bounce_cooldown = 0      # frames to ignore after bounce
vy = 0
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cx = 0
    cy = 0

    frame_index += 1
    if frame_index % 2 == 0 and i in videos60fps:
        continue

    frame = cv.resize(frame, (1280, 720))

    # predict on frame
    results = model.track(
        source=frame,
        conf=0.20,
        save=False,
        verbose=verbose
    )

    r = results[0]
    boxes = r.boxes

    best_box = None
    best_conf = 0

    # select highest confidence box
    for box in boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    # if a ball was detected
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        coordinates.append([x1, y1, x2, y2])
        
        # draw bounding box
        cv.rectangle(
            frame,
            (x1 + 5, y1 + 5),
            (x2 + 5, y2 + 5),
            (0, 255, 0),
            3
        )

        # compute center
        cx = (x1 + x2) // 2
        cy = y2

        trail.append((cx, cy))
        if len(trail) > MAX_TRAIL_LENGTH:
            trail.pop(0)

    # draw trail (old → red, new → green)
    for idx, (tx, ty) in enumerate(trail):
        alpha = idx / len(trail)
        r = int(255 * (1 - alpha))
        b = int(255 * (1 - alpha))
        g = int(255 * alpha)
        color = (b, g, 0)

        cv.circle(frame, (tx, ty), round(idx/6)+1, color, -1)

    # keypoint detections - on hold for now

    # frame = cv.resize(frame, (1280, 720)) # resize frame
    # blurred_frame = cv.GaussianBlur(frame, (5, 5), 0) # gaussian blur to clear noise

    # # convert to hsv <- low saturation = white = court lines
    # hsv_frame = cv.cvtColor(blurred_frame, cv.COLOR_BGR2HSV)

    # # create mask
    # lower = np.array([0, 0, 150])  # H, S, V
    # upper = np.array([180, 40, 255]) 
    # mask = cv.inRange(hsv_frame, lower, upper)

    # # apply mask
    # masked_frame = cv.bitwise_and(hsv_frame, hsv_frame, mask=mask)

    # # morphology
    # kernel_open = cv.getStructuringElement( # opening kernel for discarding non-lines
    #     cv.MORPH_ELLIPSE,
    #     (3,3) # change to 3x3 or 7x7 based on performance
    # )

    # mask_opened = cv.morphologyEx( # apply to mask
    #     mask,              
    #     cv.MORPH_OPEN,     
    #     kernel_open,
    # )

    # kernel_close = cv.getStructuringElement( # create closing kernel for filling lines
    #     cv.MORPH_RECT,     
    #    (5, 5) # change to 3x3 or 7x7 based on performance
    # )
    
    # mask_clean = cv.morphologyEx( # apply opening and closing
    #     mask_opened,       
    #     cv.MORPH_CLOSE,    
    #     kernel_close,
    # )

    # # apply edge detection
    # edges = cv.Canny(
    #     mask_clean,
    #     threshold1=45,
    #     threshold2=110,
    #     apertureSize=3
    # )

    # # use hough transform
    # lines = cv.HoughLinesP(
    #     edges,          
    #     rho=1,          
    #     theta=np.pi/180,
    #     threshold=100,   
    #     minLineLength=60,
    #     maxLineGap=50,  
    # )

    # draw lines
    # if lines is not None:
    #     for x1, y1, x2, y2 in lines[:,0]:
    #         cv.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)

    # frame_h, frame_w = frame.shape[:2]

    # court_offset_x = frame_w - padded_court_width
    # court_offset_y = 0

    # frame[
    #     court_offset_y:court_offset_y + padded_court_height,
    #     court_offset_x:court_offset_x + padded_court_width
    # ] = padded_court


    # only compute homography if user clicked all 4 points
    # if len(clicked_points) == 4:
        # src/dst points for mapping court points into minimap points
        # src_pts = np.array(clicked_points, dtype=np.float32)
        # dst_pts = np.array([
        #     [padding, padding],                               # top-left
        #     [padding + court_width - 1, padding],            # top-right
        #     [padding + court_width - 1, padding + court_height - 1], # bottom-right
        #     [padding, padding + court_height - 1]            # bottom-left
        # ], dtype=np.float32)

        # compute homography matrix between court and minimap
        # H, _ = cv.findHomography(src_pts, dst_pts)

        # only update ball history if we have a detection
        # if best_box is not None:
        #     cx = (x1 + x2) // 2
        #     cy = y2

        #     # append only real detections
        #     ball_history.append((cx, cy))
        #     if len(ball_history) > 3:
        #         ball_history.pop(0)

        #     # bounce detection: local minimum in y
        #     if len(ball_history) == 3 and bounce_cooldown == 0:
        #         (_, y0), (_, y1), (_, y2) = ball_history

        #         # bounce occurs when middle point is lowest
        #         if y0 > y1 and y2 > y1:
        #             bounce_x, bounce_y = ball_history[1]

                    # project bounce point once
                    # pt = np.array([[[bounce_x, bounce_y]]], dtype=np.float32)
                    # projected = cv.perspectiveTransform(pt, H)

                    # mini_x, mini_y = projected[0, 0]
                    # bounce_points.append((int(mini_x), int(mini_y)))
                    # bounce_points.append(bounce_x, bounce_y)

                    # # short cooldown to avoid double-counting
                    # bounce_cooldown = 8

        # only update ball history if we have a detection
        if best_box is not None:
            cx = (x1 + x2) // 2
            cy = y2

            # append only real detections
            ball_history.append((cx, cy))
            if len(ball_history) > 3:
                ball_history.pop(0)
            print(ball_history)
            # bounce detection: local minimum in y
            if len(ball_history) == 3 and bounce_cooldown == 0:
                (_, y0), (_, y1), (_, y2) = ball_history

                # bounce occurs when middle point is lowest
                if y0 > y1 and y2 > y1:
                    bounce_x, bounce_y = ball_history[1]

                    bounce_points.append(bounce_x, bounce_y)

                    # short cooldown to avoid double-counting
                    bounce_cooldown = 8

        # countdown cooldown
        if bounce_cooldown > 0:
            bounce_cooldown -= 1

        # draw all bounce points on minimap
        for x, y in bounce_points:
            # cv.circle(
            #     padded_court,
            #     (x, y),
            #     8,
            #     (np.random.randint(50,230), np.random.randint(50,230), np.random.randint(50,230)),
            #     -1
            # )
            cv.circle(
                frame,
                (x, y),
                8,
                (np.random.randint(50,230), np.random.randint(50,230), np.random.randint(50,230)),
                -1
            )

        # overlay minimap on frame
        # frame[court_offset_y:court_offset_y + padded_court_height,
        #     court_offset_x:court_offset_x + padded_court_width] = padded_court

    cy_prev = cy
    vy_prev = vy

    cv.imshow("frame", frame)
    cv.waitKey(1)

    # write frame
    # out.write(cv.resize(frame))

cap.release()
out.release()