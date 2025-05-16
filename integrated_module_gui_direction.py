import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from djitellopy import Tello
import time
import matplotlib.pyplot as plt
from collections import deque
import threading

# ---------- INIT ----------
def init():
    pygame.init()
    pygame.display.set_mode((400, 400))
`
def getKey(keyName):
    pygame.event.pump()
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    return keyInput[myKey]

def getUltrasonicDistance():
    try:
        distance = tello.get_distance_tof()
        return distance if distance > 0 else None
    except:
        return None

def getKeyboardInput(front_distance, down_distance):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey("a"): lr = -speed
    elif getKey("d"): lr = speed
    if getKey("w") and (front_distance is None or front_distance > 50): fb = speed
    elif getKey("s"): fb = -speed
    if getKey("UP") and (down_distance is None or down_distance > 30): ud = speed
    elif getKey("DOWN") and (down_distance is None or down_distance > 10): ud = -speed
    if getKey("LEFT"): yv = -speed
    elif getKey("RIGHT"): yv = speed

    if getKey("e"): tello.takeoff()
    if getKey("l"): tello.land()

    return [lr, fb, ud, yv]

def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width if width_in_frame > 0 else 600

def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None

# ---------- PLOTTING THREAD ----------
def live_plotting():
    plt.ion()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    line, = ax.plot([], [], 'lime', linewidth=2)
    ax.set_ylim(0, 200)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Frame Count', color='white')
    ax.set_ylabel('Ultrasonic Distance (cm)', color='white')
    ax.set_title('Live Downward Distance from Tello', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    while True:
        if len(distance_history) > 1:
            line.set_data(time_history, distance_history)
            ax.set_xlim(max(0, time_history[0]), max(50, time_history[-1]))
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.05)

# ---------- MAIN SETUP ----------
init()
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery: {tello.get_battery()}%")

yolo_model = YOLO("yolov8n.pt")

# SSD MobileNet
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().strip().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
ssd_net = cv2.dnn_DetectionModel(weightsPath, configPath)
ssd_net.setInputSize(320, 320)
ssd_net.setInputScale(1.0 / 127.5)
ssd_net.setInputMean((127.5, 127.5, 127.5))
ssd_net.setInputSwapRB(True)

KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 20.0
focal_length = 600

frame_read = tello.get_frame_read()

distance_history = deque(maxlen=50)
time_history = deque(maxlen=50)
frame_counter = 0
threading.Thread(target=live_plotting, daemon=True).start()

drone_pos = [0, 0]

# Estimate initial focal length
while True:
    frame = frame_read.frame
    results = yolo_model(frame)
    boxes = results[0].boxes
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
        focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, x2 - x1)
        break
    time.sleep(0.1)

# ---------- MAIN LOOP ----------
while True:
    frame = frame_read.frame
    front_distance = 100

    results = yolo_model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        box_width = x2 - x1
        front_distance = estimate_distance(focal_length, KNOWN_WIDTH, box_width)

        if front_distance and front_distance < 30:
            cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            tello.send_rc_control(0, -30, 0, 0)
            time.sleep(0.2)
            continue  # Skip rest of loop
        break

    classIds, confs, bbox = ssd_net.detect(frame, confThreshold=0.6, nmsThreshold=0.2)
    if isinstance(classIds, np.ndarray):
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box
            label = classNames[classId - 1].upper()
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {round(conf*100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Front Distance: {round(front_distance, 1)}cm', (x + w - 150, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 105, 180), 2)

    down_distance = getUltrasonicDistance()
    if down_distance is not None:
        distance_history.append(down_distance + 30)
        time_history.append(frame_counter)
        frame_counter += 1
        if down_distance < 30:
            cv2.putText(frame, "WARNING: MOVE UP!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    vals = getKeyboardInput(front_distance, down_distance)
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    if vals[1] > 0: drone_pos[1] += 1
    elif vals[1] < 0: drone_pos[1] -= 1
    if vals[0] > 0: drone_pos[0] += 1
    elif vals[0] < 0: drone_pos[0] -= 1

    # ----------- GUI MAP -----------
    map_display = np.zeros((400, 400, 3), dtype=np.uint8)
    center_x, center_y = 200, 200
    draw_x = center_x + drone_pos[0] * 5
    draw_y = center_y - drone_pos[1] * 5

    cv2.line(map_display, (0, 200), (400, 200), (255, 255, 255), 1)
    cv2.line(map_display, (200, 0), (200, 400), (255, 255, 255), 1)

    cv2.putText(map_display, "Left", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(map_display, "Right", (350, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(map_display, "Forward", (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(map_display, "Backward", (160, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for i in range(-8, 9):
        if i == 0:
            continue
        tick_x = center_x + i * 5
        cv2.line(map_display, (tick_x, center_y - 3), (tick_x, center_y + 3), (180, 180, 180), 1)
        tick_y = center_y + i * 5
        cv2.line(map_display, (center_x - 3, tick_y), (center_x + 3, tick_y), (180, 180, 180), 1)

    cv2.circle(map_display, (draw_x, draw_y), 5, (0, 255, 0), -1)
    cv2.putText(map_display, f"({drone_pos[0]}, {drone_pos[1]})", (draw_x + 10, draw_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"Battery: {tello.get_battery()}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Tello Feed", frame)
    cv2.imshow("Drone Coordinate Map", map_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
