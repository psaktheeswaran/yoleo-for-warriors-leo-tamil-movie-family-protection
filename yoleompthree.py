import cv2
import numpy as np
import pygame
import time
import csv
from datetime import datetime

# YOLOv3-tiny configuration and weights files
yolo_cfg = 'face-yolov3-tiny.cfg'
yolo_weights = 'face-yolov3-tiny_41000.weights'
coco_names = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
classes = []

# Load class names
with open(coco_names, 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize Pygame for playing MP3
pygame.mixer.init()
pygame.mixer.music.load('hy.mp3')

# Webcam capture
cap = cv2.VideoCapture(2)

# Initialize variables
detection_started = False
start_time = None
last_detection_time = None
stop_mp3 = True

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    if not detection_started:
        start_time = time.time()
        detection_started = True

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        last_detection_time = time.time()
        if stop_mp3:
            pygame.mixer.music.play()
            stop_mp3 = False

        with open('detections.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in indices:
                i = i
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                timestamp = datetime.now().strftime('%H:%M %p %A, %d %B %Y (%Z)')
                writer.writerow([timestamp, label, x, y, x + w, y + h])

    if time.time() - last_detection_time > 1.3:
        if not stop_mp3:
            pygame.mixer.music.stop()
            stop_mp3 = True

    if time.time() - start_time > 1.3:
        detection_started = False

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

