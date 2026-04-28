import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# -----------------------------
# Load model
# -----------------------------
model = load_model("best_model.h5", compile=False)
print("Model loaded!")

# -----------------------------
# Load car PNG
# -----------------------------
car_img = cv2.imread("car_transparent_small.png", cv2.IMREAD_UNCHANGED)
if car_img is not None:
    car_img = cv2.resize(car_img, (60, 120))
else:
    print("Warning: car image not found, skipping overlay")

# -----------------------------
# Load predictions_vs_actual.csv
# (already has predicted + actual + error from Step 5)
# -----------------------------
df = pd.read_csv("predictions_vs_actual.csv")
print(f"Total samples: {len(df)}")

# -----------------------------
# Filter by error threshold
# -----------------------------
GOOD_THRESHOLD   = 0.15   # error < 0.15  = GOOD
OK_THRESHOLD     = 0.30   # error < 0.30  = OK
HIGH_THRESHOLD   = 0.50   # error > 0.50  = HIGH ERROR

# Separate into buckets
df_good = df[np.abs(df["error"]) < GOOD_THRESHOLD]
df_ok   = df[(np.abs(df["error"]) >= GOOD_THRESHOLD) &
             (np.abs(df["error"]) < OK_THRESHOLD)]
df_high = df[np.abs(df["error"]) >= HIGH_THRESHOLD]

print(f"GOOD samples  (error < {GOOD_THRESHOLD}): {len(df_good)}")
print(f"OK samples    (error < {OK_THRESHOLD}):   {len(df_ok)}")
print(f"HIGH samples  (error > {HIGH_THRESHOLD}): {len(df_high)}")

# -----------------------------
# Build ordered playlist:
# Show more good, fewer high error
# -----------------------------
n_good = min(30, len(df_good))   # show up to 30 good
n_ok   = min(10, len(df_ok))     # show up to 10 ok
n_high = min(5,  len(df_high))   # show only 5 high error

playlist = pd.concat([
    df_good.sample(n_good, random_state=42),
    df_ok.sample(n_ok,     random_state=42),
    df_high.sample(n_high, random_state=42)
]).reset_index(drop=True)

print(f"\nPlaylist: {n_good} GOOD + {n_ok} OK + {n_high} HIGH = {len(playlist)} frames")
print("Order: Good predictions first, high errors last\n")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_for_prediction(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    img = img[int(h*0.35):int(h*0.9), :, :]
    img = cv2.resize(img, (200, 66))
    img = img.astype(np.float32) / 255.0
    return img

# -----------------------------
# Overlay car
# -----------------------------
def overlay_car(frame, car, pos=(270, 300), angle=0):
    if car is None:
        return frame
    M = cv2.getRotationMatrix2D(
        (car.shape[1]//2, car.shape[0]//2), -angle*50, 1
    )
    rotated_car = cv2.warpAffine(
        car, M, (car.shape[1], car.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    y1, y2 = pos[1], pos[1] + rotated_car.shape[0]
    x1, x2 = pos[0], pos[0] + rotated_car.shape[1]
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame
    alpha_car = rotated_car[:, :, 3] / 255.0
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha_car * rotated_car[:, :, c] +
            (1 - alpha_car) * frame[y1:y2, x1:x2, c]
        )
    return frame

# -----------------------------
# Draw HUD
# -----------------------------
def draw_hud(frame, predicted, actual, error, frame_idx, total, section_label):

    # Steering arrows
    cx, cy = 300, 250
    length = 100

    # Predicted arrow (red)
    end_x_pred = int(cx + predicted * length)
    cv2.arrowedLine(frame, (cx, cy), (end_x_pred, cy),
                    (0, 0, 255), 3, tipLength=0.3)

    # Actual arrow (yellow)
    end_x_actual = int(cx + actual * length)
    cv2.arrowedLine(frame, (cx, cy+22), (end_x_actual, cy+22),
                    (0, 255, 255), 2, tipLength=0.3)

    # Error color + label
    if error < GOOD_THRESHOLD:
        color = (0, 255, 0)
        label = "GOOD"
    elif error < OK_THRESHOLD:
        color = (0, 165, 255)
        label = "OK"
    else:
        color = (0, 0, 255)
        label = "HIGH ERROR"

    # Background box for text
    cv2.rectangle(frame, (5, 5), (280, 145), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (280, 145), (80, 80, 80), 1)

    # Text
    cv2.putText(frame, f"Predicted : {predicted:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(frame, f"Actual    : {actual:.3f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Error     : {error:.3f}", (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, label, (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Section label (top right)
    cv2.rectangle(frame, (400, 5), (635, 35), (0, 0, 0), -1)
    cv2.putText(frame, section_label, (405, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Frame counter
    cv2.putText(frame, f"Frame {frame_idx+1}/{total}", (480, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Steering angle bucket
    if actual < -0.5:
        bucket = "Sharp Left"
    elif actual < 0:
        bucket = "Gentle Left"
    elif actual == 0:
        bucket = "Straight"
    elif actual < 0.5:
        bucket = "Gentle Right"
    else:
        bucket = "Sharp Right"

    cv2.putText(frame, f"Turn: {bucket}", (480, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Legend
    cv2.rectangle(frame, (5, 448), (350, 475), (0, 0, 0), -1)
    cv2.putText(frame, "RED=Predicted  YELLOW=Actual",
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1)

    return frame

# -----------------------------
# Section separator screen
# -----------------------------
def show_section_screen(window_name, title, color, count, wait_ms=2000):
    screen = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(screen, title, (160, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(screen, f"Showing {count} samples", (180, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(screen, "Press any key to start", (160, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
    cv2.imshow(window_name, screen)
    cv2.waitKey(0)

# -----------------------------
# Main visualization loop
# -----------------------------
cv2.namedWindow("Self-Driving Visualization", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Self-Driving Visualization", 640, 480)

print("Controls: any key = next | 'q' = quit | 'r' = restart | 's' = skip section")

sections = [
    ("GOOD Predictions",  df_good.sample(n_good, random_state=42).reset_index(drop=True), (0, 255, 0)),
    ("OK Predictions",    df_ok.sample(n_ok,     random_state=42).reset_index(drop=True), (0, 165, 255)),
    ("HIGH ERROR Cases",  df_high.sample(n_high, random_state=42).reset_index(drop=True), (0, 0, 255)),
]

total = n_good + n_ok + n_high
global_idx = 0

for section_name, section_df, section_color in sections:

    # Show section intro screen
    show_section_screen(
        "Self-Driving Visualization",
        section_name,
        section_color,
        len(section_df)
    )

    skip_section = False
    i = 0

    while i < len(section_df) and not skip_section:
        row    = section_df.iloc[i]
        path   = row["image_path"]
        actual = float(row["actual_steering"])
        error  = abs(float(row["error"]))

        frame = cv2.imread(path)
        if frame is None:
            print(f"Skipping missing: {path}")
            i += 1
            global_idx += 1
            continue

        frame = cv2.resize(frame, (640, 480))

        # Re-predict for visualization
        img_input = preprocess_for_prediction(frame)
        predicted = model.predict(
            np.expand_dims(img_input, axis=0), verbose=0
        )[0][0]

        # Overlay
        frame = overlay_car(frame, car_img, pos=(290, 300), angle=predicted)
        frame = draw_hud(
            frame, predicted, actual, error,
            global_idx, total, section_name
        )

        cv2.imshow("Self-Driving Visualization", frame)
        key = cv2.waitKey(0)

        if key == ord('q'):
            print("Quit by user")
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            print(f"Skipping section: {section_name}")
            skip_section = True
        elif key == ord('r'):
            print("Restarting section...")
            i = 0
            continue

        i += 1
        global_idx += 1

cv2.destroyAllWindows()
print("Visualization ended!")
print(f"\nSummary shown: {n_good} GOOD | {n_ok} OK | {n_high} HIGH ERROR")
