import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)

    annotated = frame.copy()

    frame_h, frame_w = frame.shape[:2]
    frame_center_x =frame_w // 2
    frame_center_y = frame_h // 2

    cv2.circle(annotated, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)

    for box  in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        label = model.names[cls]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(annotated, f"{label} {conf:.2f} ({cx},{cy})",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2 )

        error_x = cx - frame_center_x
        error_y = cy - frame_center_y

        if error_x < -50:
            direction = "Move Left"
        elif error_x > 50:
            direction = "Move Right"
        else:
            direction = "Centered"

        cv2.putText(annotated, direction, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        print(f"{label}: center=({cx}, {cy}), error_x={error_x}, error_y={error_y}")

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()