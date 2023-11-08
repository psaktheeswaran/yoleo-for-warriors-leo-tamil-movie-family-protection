import cv2
import numpy as np
import pygame
from datetime import datetime
import os
import csv
from PIL import Image

# Initialize pygame
pygame.init()
pygame.mixer.init()  # Initialize the mixer for audio playback

# Create a display window
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Object Detection")

# Load YOLO model
net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load MP3 file for audio alert
mp3_file = "hy.mp3"
audio_alert = pygame.mixer.Sound(mp3_file)  # Load the MP3 as a sound object

# Load overlay image (JPEG)
overlay = cv2.imread("dffdf.jpg")

# Create a folder for storing detected faces
current_time = datetime.now().strftime("%Y%m%d_%H%M")
output_folder = os.path.join("detected_faces", current_time)
os.makedirs(output_folder, exist_ok=True)

# Create a CSV file to store detection information with timestamp and location
csv_filename = os.path.join(output_folder, "detections.csv")
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Label", "Confidence", "X", "Y", "Width", "Height", "Timestamp", "Location"])

# Function to overlay an image on a detected object
def overlay_image(img, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    img[y:y+h, x:x+w] = overlay_resized

# Open the webcam (you can also use an RTSP stream)
cap = cv2.VideoCapture(2)  # Change the camera index to 0 for the default camera

# Create a clock object for controlling frame rate
clock = pygame.time.Clock()

# Initialize variables for audio control
audio_playing = False

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically to correct the upside-down issue
    frame = cv2.flip(frame, 0)

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    detection_present = False

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
                detection_present = True  # Detection is present

    # Check if the audio should be playing
    if detection_present:
        if not audio_playing:
            audio_alert.play()
            audio_playing = True
    else:
        audio_alert.stop()
        audio_playing = False

    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box
        class_id = class_ids[i]
        label = str(classes[class_id])
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        overlay_image(frame, overlay, x, y, w, h)

        # Save the detected face without a mask
        detected_face = frame[y:y+h, x:x+w]
        face_image = Image.fromarray(cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB))
        face_image.save(os.path.join(output_folder, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"))

        # Write the detection info to the CSV file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        location = "Madurai"
        csv_writer.writerow([label, confidence, x, y, w, h, timestamp, location])

    # Convert the frame to Pygame surface directly
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)

    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    clock.tick(30)  # Adjust the frame rate as needed

# Clean up
cap.release()
cv2.destroyAllWindows()
csv_file.close()
pygame.quit()

