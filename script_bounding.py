import cv2
import numpy as np
import time

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Start webcam
cap = cv2.VideoCapture(0)

frame_id = 0

while True:
    ret, img = cap.read()

    if not ret or img is None:
        print("Failed to grab frame")
        continue

    height, width, _ = img.shape

    # ---- YOLO ----
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # ---- NMS ----
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ---- COUNT OBJECTS ----
    counts = {}

    if len(indexes) > 0:
        for i in indexes.flatten():
            label = classes[class_ids[i]]

            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

            # draw box
            x, y, w, h = boxes[i]
            conf = confidences[i]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- PRINT SUMMARY ----
    print("\nFrame:", frame_id)

    total = 0

    if counts:
        for label, count in counts.items():
            print(f"{label}: {count}")
            total += count
        print("Total objects:", total)
    else:
        print("No objects detected")

    # ---- SAVE FRAME ----
    #filename = f"frame_{frame_id}.jpg"
    filename = f"frame_0.jpg"
    cv2.imwrite(filename, img)
    print("Saved:", filename)

    frame_id += 1

    # optional slowdown so disk isn't spammed
    time.sleep(0.1)

    # ---- EXIT ----
    if input("Press q + Enter to quit: ") == "q":
        break

cap.release()
cv2.destroyAllWindows()