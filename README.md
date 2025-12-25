# _MyProjects_
from ultralytics import YOLO
import cv2
import time

# Load YOLOv8
model = YOLO("yolov8n.pt")

vehicle_classes = ["car", "motorbike", "bus", "truck"]

video_paths = ["TrafficVideo.mp4",
    "TrafficVideo2.mp4",
    "TrafficVideo3.mp4",
    "TrafficVideo2.mp4"]
caps = [cv2.VideoCapture(vp) for vp in video_paths]

BASE_GREEN = 10
MAX_GREEN = 40
YELLOW_TIME = 3
RED_TIME = 5

lane_status = ["RED", "RED", "RED", "RED"]
current_lane = 0
last_switch_time = time.time()
green_duration = BASE_GREEN
time_remaining = green_duration  # countdown tracker

def count_vehicles(frame):
    results = model(frame, verbose=False)
    count = 0
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls[0])]
            if cls_name in vehicle_classes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, cls_name, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return count, frame

while True:
    frames, counts = [], []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        frame = cv2.resize(frame, (480, 360))
        count, processed_frame = count_vehicles(frame)
        counts.append(count)

        cv2.putText(processed_frame, f"Vehicles: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        frames.append(processed_frame)

    # Calculate elapsed time for current lane
    elapsed = time.time() - last_switch_time
    if lane_status[current_lane] == "GREEN":
        time_remaining = max(0, green_duration - int(elapsed))
        if elapsed >= green_duration:
            lane_status[current_lane] = "YELLOW"
            last_switch_time = time.time()
    elif lane_status[current_lane] == "YELLOW":
        time_remaining = max(0, YELLOW_TIME - int(elapsed))
        if elapsed >= YELLOW_TIME:
            lane_status[current_lane] = "RED"
            current_lane = (current_lane + 1) % 4
            green_duration = min(BASE_GREEN + counts[current_lane]*2, MAX_GREEN)
            lane_status[current_lane] = "GREEN"
            last_switch_time = time.time()
    elif lane_status[current_lane] == "RED":
        time_remaining = max(0, RED_TIME - int(elapsed))
        if elapsed >= RED_TIME and lane_status[current_lane] != "GREEN":
            green_duration = min(BASE_GREEN + counts[current_lane]*2, MAX_GREEN)
            lane_status[current_lane] = "GREEN"
            last_switch_time = time.time()

    for i, f in enumerate(frames):
        status = lane_status[i]

        if i == current_lane:
            countdown = time_remaining
        else:
            countdown = 0

        color = (0,0,255) if status=="RED" else (0,255,0) if status=="GREEN" else (0,255,255)
        symbol = "🔴" if status=="RED" else "🟢" if status=="GREEN" else "🟡"

        # ✅ FIX: No "???" anymore
        if status in ["GREEN", "YELLOW"]:
            label = f"Lane {i+1}: {status} {symbol} ({countdown}s left)"
        else:  # RED → no timer shown
            label = f"Lane {i+1}: {status} {symbol}"

        cv2.putText(f, label, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    # Arrange 4 videos into a grid (2x2)
    top = cv2.hconcat([frames[0], frames[1]])
    bottom = cv2.hconcat([frames[2], frames[3]])
    grid = cv2.vconcat([top, bottom])

    cv2.imshow("Smart Traffic Management - YOLOv8", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
