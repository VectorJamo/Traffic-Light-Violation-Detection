import cv2
import os
from ultralytics import YOLO
import numpy as np
import time

# ---------------------------
# REGION DEFINITIONS
# ---------------------------
# Areas of interest location in pixel - coordinates
RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]]) # Vehicle pass intersection location

# ---------------------------
# LOAD YOLO
# ---------------------------
model = YOLO("yolov8m.pt")
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck"]
model = YOLO("yolov8m.pt") # Load the YOLO Model

coco = model.model.names # Returns a dict of object IDs(Object Classes) to Object Names

# ---------------------------
# SPEED TRACKING CONFIG
# ---------------------------
DISTANCE_METERS = 5  # Real-world distance between the two lines
LINE_A = 200          # y-coordinate of the first horizontal line
LINE_B = 270       # y-coordinate of the second horizontal line
SPEED_LIMIT = 50      # km/h

vehicle_id_counter = 0
active_vehicles = {}       # Current vehicle centers {id: (cx, cy)}
vehicle_times = {}         # Crossing times {id: {"A": timestamp, "B": timestamp}}
vehicle_speeds = {}        # Calculated speed {id: speed_kph}

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def match_vehicle(cx, cy, active, threshold=60):
    for vid, (px, py) in active.items():
        if np.linalg.norm([cx - px, cy - py]) < threshold:
            return vid
    return None

def draw_text_with_bg(frame, txt, pos, color=(255,255,255)):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Checks if a traffic light (red/green) is ON.
def is_region_light(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts frame to grayscale

    mask = np.zeros_like(gray_image) # Mask is a black image the same size as gray_image.
    
    cv2.fillPoly(mask, [np.array(polygon)], 255) # Draw a white polygon where the traffic light is located.
    
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask) # Keeps only the pixels inside the polygon. 
    # Uses the bitwise operation which is more efficient than just subtracting the pixel values
    
    mean_brightness = cv2.mean(roi, mask=mask)[0] # Calculate the average brightness of pixels in the masked region.
    
    return mean_brightness > brightness_threshold # mean brightness is higher than the threshold means light is ON

def is_region_light(image, polygon, brightness_threshold=130):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [polygon], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.mean(roi, mask=mask)[0] > brightness_threshold

# ---------------------------
# MAIN VIDEO LOOP
# ---------------------------
cap = cv2.VideoCapture("videos/traffic.mp4")
FPS = 30
FRAME_TIME = 1.0 / FPS

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1100, 700))

    # Draw horizontal reference lines
    cv2.line(frame, (0, LINE_A), (frame.shape[1], LINE_A), (255,0,0), 2)
    cv2.line(frame, (0, LINE_B), (frame.shape[1], LINE_B), (0,0,255), 2)

    # YOLO detection
    results = model.predict(frame, conf=0.75)
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            label = coco[int(cls)]
            if label not in TargetLabels:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2

            # Track vehicle ID
            vid = match_vehicle(cx, cy, active_vehicles)
            if vid is None:
                vehicle_id_counter += 1
                vid = vehicle_id_counter

            active_vehicles[vid] = (cx, cy)

            # Initialize crossing times
            if vid not in vehicle_times:
                vehicle_times[vid] = {"A": None, "B": None}

            now = time.time()
            # Line A crossing
            if vehicle_times[vid]["A"] is None and cy >= LINE_A:
                vehicle_times[vid]["A"] = now
            # Line B crossing
            elif vehicle_times[vid]["A"] is not None and vehicle_times[vid]["B"] is None and cy >= LINE_B:
                vehicle_times[vid]["B"] = now
                dt = vehicle_times[vid]["B"] - vehicle_times[vid]["A"]
                if dt > 0:
                    speed_mps = DISTANCE_METERS / dt
                    speed_kph = speed_mps * 3.6
                    vehicle_speeds[vid] = speed_kph

            # Draw bounding box
            color = (0,255,0)
            if vid in vehicle_speeds and vehicle_speeds[vid] > SPEED_LIMIT:
                color = (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw speed text
            if vid in vehicle_speeds:
                draw_text_with_bg(frame, f"{vehicle_speeds[vid]:.1f} km/h", (x1, y1-10), color)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        break

    # Maintain FPS
    elapsed = time.time() - start_time
    if elapsed < FRAME_TIME:
        time.sleep(FRAME_TIME - elapsed)

# Load the video file
cap = cv2.VideoCapture("videos/tr.mp4")

# While the video file is open, we wish to perform calculations on each frame
while cap.isOpened():
    success, frame = cap.read() # Get the video frame
    if not success:
        print("number of frames have finished.")
        break
    else:
        frame = cv2.resize(frame, (1100, 700)) 
        # Draw the regions of interest locations
        cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1) 
        cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
        cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)
        
        # Do the actual prediction
        results = model.predict(frame, conf=0.75)
        for result in results:
            boxes = result.boxes.xyxy # Get the bounding boxes of the objects detected
            confs = result.boxes.conf # Get the confidence score for those objects
            classes = result.boxes.cls # Get the name of the objects

            # Iterate over all the boxes and draw them
            for box, conf, cls in zip(boxes, confs, classes):
                if coco[int(cls)] in TargetLabels: # Check if the detected object is within interested labels and draw outline around them
                    x, y, w, h = box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(frame, (x, y), (w, h), [0, 255, 0], 2)
                    draw_text_with_background(frame, 
                                      f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 0, 255))  # Red border
                # Check if its red light.
                if is_region_light(frame, RedLight):
                    # Check if vehicle is inside the monitored road area (ROI).
                    if cv2.pointPolygonTest(ROI, (x, y), False) >= 0 or cv2.pointPolygonTest(ROI, (w, h), False) >= 0:
                        draw_text_with_background(frame, 
                                      f"The {coco[int(cls)].capitalize()} violated the traffic signal.", 
                                      (10, 30), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),  # White text
                                      (0, 0, 0),  # Black background
                                      (0, 0, 255))  # Red border

                        cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)
                        cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 2)
                        # time.sleep(1)
    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
