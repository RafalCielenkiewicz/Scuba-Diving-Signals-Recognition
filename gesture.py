import torch
import cv2
import numpy as np
from tensorflow.keras.models import load_model

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp13/weights/best.pt')
gesture_model = load_model('gesture_classification_model.keras')
gesture_classes = ["down", "level", "low", "ok", "stop", "up"]

yolo_conf_threshold = 0.5
gesture_conf_threshold = 0.6

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = yolo_model(frame)
    detections = results.xyxy[0]

    for *xyxy, conf, cls in detections.tolist():
        if conf < yolo_conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        hand_region = frame[y1:y2, x1:x2]

        if hand_region.size == 0:
            continue

        hand_region_resized = cv2.resize(hand_region, (150, 150))
        hand_region_normalized = hand_region_resized / 255.0
        hand_region_input = np.expand_dims(hand_region_normalized, axis=0)

        gesture_pred = gesture_model.predict(hand_region_input, verbose=0)
        max_confidence = np.max(gesture_pred)
        gesture_label = gesture_classes[np.argmax(gesture_pred)]

        if max_confidence < gesture_conf_threshold:
            gesture_label = "Uncertain"

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"{gesture_label} ({max_confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Hand Detection and Gesture Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

