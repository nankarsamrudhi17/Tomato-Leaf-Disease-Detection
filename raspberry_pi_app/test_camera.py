"""
Quick webcam test. Run this BEFORE app.py.

    python3 test_camera.py

Shows a live window (press Q to quit) and saves test_frame.jpg.
Prints resolution, brightness and sharpness so you can tune the
thresholds at the top of app.py.
"""
import sys

import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(f"Cannot open camera index {CAMERA_INDEX}. Try 1, or check the USB connection (ls /dev/video*).")
    sys.exit(1)

for _ in range(10):          # let auto-exposure settle
    ok, frame = cap.read()

if not ok:
    print("Camera opened but no frame received.")
    sys.exit(1)

cv2.imwrite("test_frame.jpg", frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print("Resolution :", frame.shape[1], "x", frame.shape[0])
print("Brightness :", round(float(gray.mean()), 1), "(app rejects < 50 or > 220)")
print("Sharpness  :", round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1), "(app rejects < 60)")
print("Saved test_frame.jpg")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Camera test - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
except cv2.error:
    print("(No display available - skipped live window.)")

cap.release()
cv2.destroyAllWindows()
