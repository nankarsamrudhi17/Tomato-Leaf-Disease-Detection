"""
Collect your own webcam images for training (run on the Raspberry Pi or a PC).

Controls (click the video window first):
    N / P   next / previous class
    SPACE   save one image
    A       toggle AUTO mode (saves one image every 1.5 s - move/rotate the leaf between shots)
    Q       quit

Images go to:  webcam_images/<class_name>/<class_name>_<timestamp>.jpg
Class folder names must match the ones used in training (see CLASSES below).

Tips for good data
  * Same setup you will use in the final app: same camera, distance (15-25 cm), light.
  * Vary angle, position and lighting a little. Use 50-100+ images per class.
  * For Not_a_leaf: hands, table, wall, soil, empty frame, other objects/plants.
  * Take photos of MANY different leaves, not 100 shots of one leaf.
"""
import os
import time

import cv2

CAMERA_INDEX = 0
SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_images")
AUTO_INTERVAL = 1.5          # seconds between auto-saves
BLUR_THRESHOLD = 60.0        # auto mode skips images blurrier than this

CLASSES = [
    "Not_a_leaf",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Mosaic_virus",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
]


def count_images(cls):
    d = os.path.join(SAVE_ROOT, cls)
    return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")]) if os.path.isdir(d) else 0


def save_frame(frame, cls):
    d = os.path.join(SAVE_ROOT, cls)
    os.makedirs(d, exist_ok=True)
    name = f"{cls}_{int(time.time() * 1000)}.jpg"      # timestamp keeps files in capture order
    cv2.imwrite(os.path.join(d, name), frame)


def sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Camera not found. Check the USB webcam or change CAMERA_INDEX.")

    idx, auto, last_auto, flash_until = 0, False, 0.0, 0.0
    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost camera signal.")
            break

        cls = CLASSES[idx]
        now = time.time()
        if auto and now - last_auto >= AUTO_INTERVAL:
            last_auto = now
            if sharpness(frame) >= BLUR_THRESHOLD:
                save_frame(frame, cls)
                flash_until = now + 0.2

        view = frame.copy()
        color = (0, 200, 0) if now < flash_until else (255, 255, 255)
        cv2.rectangle(view, (0, 0), (view.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(view, f"[{idx + 1}/{len(CLASSES)}] {cls}   saved: {count_images(cls)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(view, f"SPACE=save  A=auto({'ON' if auto else 'off'})  N/P=class  Q=quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Collect leaf images", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            save_frame(frame, cls)
            flash_until = time.time() + 0.2
        elif key == ord("a"):
            auto = not auto
            last_auto = time.time()
        elif key == ord("n"):
            idx = (idx + 1) % len(CLASSES)
        elif key == ord("p"):
            idx = (idx - 1) % len(CLASSES)

    cap.release()
    cv2.destroyAllWindows()
    print("\nImages per class:")
    for c in CLASSES:
        print(f"  {c:32s} {count_images(c)}")


if __name__ == "__main__":
    main()
