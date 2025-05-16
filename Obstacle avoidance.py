import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from djitellopy import Tello
import time
import matplotlib.pyplot as plt
from collections import deque
import threading

# Initialize Pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((400, 400))

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

    if getKey("a"):
        lr = -speed
    elif getKey("d"):
        lr = speed

    if getKey("w") and (front_distance is None or front_distance > 50):
        fb = speed
    elif getKey("s"):
        fb = -speed

    if getKey("UP") and (down_distance is None or down_distance > 30):
        ud = speed
    elif getKey("DOWN") and (down_distance is None or down_distance > 10):
        ud = -speed

    if getKey("LEFT"):
        yv = -speed
    elif getKey("RIGHT"):
        yv = speed

    if getKey("e"):
        print("Taking off...")
        tello.takeoff()
        time.sleep(2)

    if getKey("l"):
        print("Landing...")
        tello.land()
        time.sleep(2)

    return [lr, fb, ud, yv]

def calculate_focal_length(known_distance, real_width, width_in_frame):
    return (width_in_frame * known_distance) / real_width if width_in_frame > 0 else 600

def estimate_distance(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame if width_in_frame > 0 else None

# Thread function for live plotting
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

    global distance_history, time_history
    while True:
        if len(distance_history) > 1:
            line.set_data(time_history, distance_history)
            ax.set_xlim(max(0, time_history[0]), max(50, time_history[-1]))
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.05)

# ============ Main Code Starts Here =============

# Initialize Pygame
init()

# Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()

battery_level = tello.get_battery()
print(f"Battery Level: {battery_level}%")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Constants for distance calculation
KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 20.0

# Setup live plotting variables
distance_history = deque(maxlen=50)
time_history = deque(maxlen=50)
frame_counter = 0

# Start live plotting in another thread
plot_thread = threading.Thread(target=live_plotting, daemon=True)
plot_thread.start()

# Ensure valid frame
frame = None
while frame is None:
    frame = tello.get_frame_read().frame
    time.sleep(0.1)

# Estimate initial focal length
results = model(frame)
objects = results[0].boxes

if len(objects) > 0:
    first_box = objects.xyxy[0]
    box_width = first_box[2] - first_box[0]
    focal_length = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, box_width)
    print(f"Estimated Focal Length: {focal_length:.2f} cm")
else:
    focal_length = 600

# Main Loop
while True:
    frame = tello.get_frame_read().frame
    if frame is None:
        continue

    results = model(frame)
    objects = results[0].boxes
    front_distance = 100

    for box in objects.xyxy:
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        front_distance = estimate_distance(focal_length, KNOWN_WIDTH, box_width)

        if front_distance:
            cv2.putText(frame, f"Front Distance: {front_distance:.2f} cm", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if front_distance < 20:
                print("WARNING: Object is very close! Moving backward...")
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                tello.send_rc_control(0, -30, 0, 0)
                time.sleep(0.1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    down_distance = getUltrasonicDistance()
    print(f"Ultrasonic (Ground) Distance: {down_distance if down_distance is not None else 'N/A'} cm")

    if down_distance is not None:
        distance_history.append(down_distance+30)
        time_history.append(frame_counter)
        frame_counter += 1

    if down_distance is not None and down_distance < 30:
        print("WARNING: Too close to ground! Moving up...")
        cv2.putText(frame, "WARNING: MOVE UP!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # tello.send_rc_control(0, 0, 30, 0)
        time.sleep(0.1)

    vals = getKeyboardInput(front_distance, down_distance)
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    cv2.putText(frame, f"Battery: {tello.get_battery()}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Tello Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
tello.streamoff()
tello.end()
cv2.destroyAllWindows()