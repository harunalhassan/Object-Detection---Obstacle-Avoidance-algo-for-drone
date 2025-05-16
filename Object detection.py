import cv2
from djitellopy import tello
import cvzone
import pygame
from time import sleep

# Initialize pygame for keyboard control
def init():
    pygame.init()
    pygame.display.set_mode((480, 480))

def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, f'K_{keyName}')
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if getKey("LEFT"):
        lr = -speed
    elif getKey("RIGHT"):
        lr = speed

    if getKey("UP"):
        fb = speed
    elif getKey("DOWN"):
        fb = -speed

    if getKey("w"):
        ud = speed
    elif getKey("s"):
        ud = -speed

    if getKey("a"):
        yv = -speed
    elif getKey("d"):
        yv = speed

    if getKey("q"):
        me.land()
        sleep(3)

    if getKey("e"):
        me.takeoff()

    return [lr, fb, ud, yv]

# Initialize pygame
init()

# Initialize the drone
me = tello.Tello()
me.connect()
print(f"Battery: {me.get_battery()}%")
me.streamoff()
me.streamon()

# Load the object detection model
thres = 0.6
nmsThres = 0.2
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Get keyboard inputs
    vals = getKeyboardInput()

    # Ensure the drone stays in place if no key is pressed
    if vals == [0, 0, 0, 0]:
        me.send_rc_control(0, 0, 0, 0)
    else:
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the drone's camera feed
    img = me.get_frame_read().frame

    # Perform object detectione
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}%',
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    except:
        pass

    # Display the video feed
    cv2.imshow("Drone Feed", img)
    cv2.waitKey(1)
