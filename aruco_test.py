import cv2
import cv2.aruco as aruco

# Map of ArUco marker IDs -> human-readable labels.
# Edit these IDs/labels to match whichever 4 markers you print out.
MARKER_LABELS = {
    0: "Blinky",
    1: "Pinky",
    2: "Inky",
    3: "Clyde",
}

# DICT_4X4_50 is a small, easy-to-detect set with plenty of room for 4 IDs.
# Generate matching markers at https://chev.me/arucogen/ (use the same dict).
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector   = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Track which markers were visible last frame so we only print on first-sight
# (edge-triggered) instead of spamming the terminal every frame.
prev_seen = set()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    current_seen = set()
    if ids is not None:
        for marker_id in ids.flatten():
            mid = int(marker_id)
            current_seen.add(mid)
            if mid not in prev_seen:
                label = MARKER_LABELS.get(mid, "(unlabeled)")
                print(f"Detected marker {mid}: {label}")

        aruco.drawDetectedMarkers(frame, corners, ids)

    for lost in prev_seen - current_seen:
        label = MARKER_LABELS.get(lost, "(unlabeled)")
        print(f"Lost marker {lost}: {label}")

    prev_seen = current_seen

    cv2.imshow("ArUco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
