import cv2
import time
import subprocess
from ultralytics import YOLO

# ==============================
# MODEL
# ==============================

MODEL_PATH = "/Users/admin/runs/detect/train-2/weights/best.pt"

model = YOLO(MODEL_PATH)

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

window_name = "AI Worker Safety Monitoring"

# ==============================
# ALERT SETTINGS
# ==============================

last_alert_time = 0
ALERT_COOLDOWN = 5

violation_classes = {
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots"
}

normal_classes = {
    "helmet",
    "goggles",
    "gloves",
    "boots",
    "vest"
}

# ==============================
# MAIN LOOP
# ==============================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera.")
        break

    # YOLO detection
    results = model(
        frame,
        conf=0.20,
        iou=0.45,
        verbose=False
    )[0]

    workers = 0

    violation_best = {}
    normal_best = {}

    # ==============================
    # PROCESS DETECTIONS
    # ==============================

    for box in results.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Count workers
        if name == "Person":
            workers += 1

        # Keep strongest violation
        if name in violation_classes:

            if (
                name not in violation_best
                or conf > violation_best[name][0]
            ):
                violation_best[name] = (
                    conf,
                    (x1, y1, x2, y2)
                )

        # Keep strongest normal PPE
        elif name in normal_classes:

            if (
                name not in normal_best
                or conf > normal_best[name][0]
            ):
                normal_best[name] = (
                    conf,
                    (x1, y1, x2, y2)
                )

    # ==============================
    # DRAW NORMAL PPE
    # ==============================

    for name, (conf, (x1, y1, x2, y2)) in normal_best.items():

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{name} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ==============================
    # DRAW VIOLATIONS
    # ==============================

    for name, (conf, (x1, y1, x2, y2)) in violation_best.items():

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        label = f"{name} {conf:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    # ==============================
    # SAFETY ALERT
    # ==============================

    if len(violation_best) > 0:

        current_time = time.time()

        # Voice alert every 5 seconds
        if current_time - last_alert_time >= ALERT_COOLDOWN:

            subprocess.Popen([
                "say",
                "Warning! Safety violation detected!"
            ])

            last_alert_time = current_time

        # Display alert
        cv2.putText(
            frame,
            "SAFETY ALERT!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # ==============================
    # INFORMATION DISPLAY
    # ==============================

    cv2.putText(
        frame,
        f"Workers: {workers}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"No Helmet: {1 if 'no_helmet' in violation_best else 0}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"No Goggles: {1 if 'no_goggle' in violation_best else 0}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"No Gloves: {1 if 'no_gloves' in violation_best else 0}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"No Boots: {1 if 'no_boots' in violation_best else 0}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==============================
    # SHOW CAMERA
    # ==============================

    cv2.imshow(window_name, frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
