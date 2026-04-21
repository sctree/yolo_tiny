import cv2
import numpy as np
import zmq
import json
import base64
import time

# ── ZeroMQ setup ──────────────────────────────────────────────
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://0.0.0.0:5555")   # laptop connects to this

# ── Load YOLOv4-tiny ──────────────────────────────────────────
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
#net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights", "yolov4-tiny-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THRESH = 0.5
NMS_THRESH  = 0.4

# ── Transmission settings (applied AFTER inference) ───────────
# YOLO runs on the raw frame; only the outgoing image is shrunk/compressed.
DISPLAY_SCALE = 0.5   # 640x480 -> 320x240
JPEG_QUALITY  = 50

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            cid    = int(np.argmax(scores))
            conf   = float(scores[cid])
            if conf < CONF_THRESH:
                continue
            cx, cy, bw, bh = (det[0]*w, det[1]*h, det[2]*w, det[3]*h)
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(conf)
            class_ids.append(cid)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = classes[class_ids[i]]
            conf  = confidences[i]
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "box":        [x, y, bw, bh]
            })

    # ── Downscale + JPEG-encode AFTER inference ───────────────
    if DISPLAY_SCALE != 1.0:
        display_frame = cv2.resize(
            frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA
        )
    else:
        display_frame = frame

    _, buf = cv2.imencode(".jpg", display_frame,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # ── Publish both on the same socket ───────────────────────
    payload = json.dumps({
        "timestamp":  time.time(),
        "image_b64":  img_b64,
        "detections": detections
    })
    socket.send_string(payload)
    print(f"Published {len(detections)} detections")