import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("face-yolov3-tiny_41000.weights", "face-yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load JPEG image for overlay
overlay = cv2.imread("dffdf.jpg")

# Create a function to overlay an image on a detected object
def overlay_image(img, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))
    
    if overlay_resized.shape[0] == h and overlay_resized.shape[1] == w:
        img[y:y+h, x:x+w] = overlay_resized
    else:
        print("Overlay image dimensions do not match the region to be overlaid.")

# Open the webcam (you can also use an RTSP stream)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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

    # Apply NMS to the detected objects
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        class_id = class_ids[i]
        label = str(classes[class_id])
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        overlay_image(frame, overlay, x, y, w, h)

    cv2.imshow("Webcam Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

