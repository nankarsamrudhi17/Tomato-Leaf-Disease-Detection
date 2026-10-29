"""
IoT-Based Leaf Disease Detection System  -  Tkinter frontend
Runs on Raspberry Pi (or any PC with a webcam).

Files needed in the SAME folder as this script:
    model.tflite   - trained MobileNetV3-Large model (exported from Colab)
    labels.txt     - one class name per line, same order as training

If model.tflite is missing, the app starts in DEMO MODE (random fake
predictions, clearly labelled) so you can test the interface first.

Run:   python3 app.py
Keys:  F11 = toggle fullscreen,  Esc = leave fullscreen
"""

import os
import time
import datetime
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# ----------------------------------------------------------------------
# SETTINGS  (change these if needed)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(names):
    """Return the first existing file from names (in BASE_DIR), else the first name."""
    for n in names:
        p = os.path.join(BASE_DIR, n)
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, names[0])


# Accepts either the default names or the names from your Colab/Drive files
MODEL_PATH = find_file(["model.tflite", "tomato_leaf_model.tflite"])
LABELS_PATH = find_file(["labels.txt", "tomato_leaf_labels.txt"])
SAVE_DIR = os.path.join(BASE_DIR, "captures")

# Camera: 0 = laptop/USB webcam.  For a phone (IP Webcam app) use its stream URL, e.g.
# CAMERA_SOURCE = "http://10.10.15.242:8080/video"
CAMERA_SOURCE = 0
CONF_THRESHOLD = 0.80       # below this -> "not sure, retake"
MIN_LEAF_FRACTION = 0.20    # at least this share of pixels must be leaf-coloured (green/yellow/brown)
UPLOAD_MAX_SIDE = 640       # uploaded photos are shrunk to this size before analysis
BLUR_THRESHOLD = 60.0       # Laplacian variance; lower = blurrier
DARK_THRESHOLD = 50         # mean brightness below this = too dark
BRIGHT_THRESHOLD = 220      # mean brightness above this = overexposed
NOT_LEAF_NAMES = {"not_a_leaf", "not a leaf", "background", "no_leaf"}

# PREPROCESSING - MUST match how the model was trained (check your training notebook):
#   "raw"     : model expects 0-255 pixels (Keras MobileNetV3 with built-in preprocessing,
#               or a model that has a Rescaling layer inside)
#   "div255"  : model was trained with images / 255.0  (values 0-1)
#   "neg1_1"  : model was trained with (images / 127.5) - 1  (values -1 to 1,
#               e.g. MobileNetV2 preprocess_input)
# Wrong setting = wrong or always-the-same predictions. If unsure, test all three
# on one image whose class you know.
PREPROCESS = "raw"

PREVIEW_W, PREVIEW_H = 460, 345
DEMO_LABELS = ["Healthy", "Early_blight", "Late_blight", "Leaf_Mold", "Not_a_leaf"]


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def pretty(name):
    """'Tomato___Late_blight' -> 'Tomato - Late blight'"""
    name = name.replace("___", " - ").replace("_", " ")
    return name.strip()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def open_camera(source):
    """Open a USB webcam (int) or a network stream such as a phone camera (URL string)."""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # less lag on network streams
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def resize_max(frame, max_side):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def check_leaf_present(frame_bgr):
    """Rough colour test: is a good share of the image green/yellow/brown plant colour?
    The model has no 'not a leaf' class, so this stops screens, hands, walls etc.
    Returns (ok, fraction)."""
    small = cv2.resize(frame_bgr, (160, 120))
    h, s, v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
    mask = (h >= 15) & (h <= 90) & (s >= 50) & (v >= 40)
    frac = float(mask.mean())
    return frac >= MIN_LEAF_FRACTION, frac


def check_quality(frame_bgr):
    """Return (ok, message). Rejects blurry / too dark / overexposed images."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < DARK_THRESHOLD:
        return False, "Image too dark - improve the lighting and retake."
    if brightness > BRIGHT_THRESHOLD:
        return False, "Image overexposed - reduce glare and retake."
    if sharpness < BLUR_THRESHOLD:
        return False, "Image looks blurry - hold steady, adjust distance and retake."
    return True, "OK"


def load_interpreter_class():
    """Find a TFLite runtime: tflite_runtime -> ai_edge_litert -> tensorflow."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        return None


# ----------------------------------------------------------------------
# PREDICTOR
# ----------------------------------------------------------------------
class Predictor:
    def __init__(self, model_path, labels_path):
        self.demo = False
        self.warning = ""
        self.labels = []

        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = [ln.strip() for ln in f if ln.strip()]

        Interpreter = load_interpreter_class()
        if Interpreter is None or not os.path.exists(model_path):
            self.demo = True
            if not self.labels:
                self.labels = DEMO_LABELS
            self.warning = "DEMO MODE: model.tflite or TFLite runtime not found. Predictions are FAKE."
            return

        self.interp = Interpreter(model_path=model_path, num_threads=4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, _ = self.inp["shape"]

        n_out = int(self.out["shape"][-1])
        if not self.labels:
            self.labels = [f"Class {i}" for i in range(n_out)]
            self.warning = "labels.txt not found - showing class numbers."
        elif len(self.labels) != n_out:
            self.warning = (f"labels.txt has {len(self.labels)} names but model "
                            f"outputs {n_out} classes - check your files!")

    def predict(self, frame_bgr):
        """Return probability array (one value per class)."""
        if self.demo:
            return np.random.dirichlet(np.ones(len(self.labels)) * 0.5)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)      # OpenCV is BGR
        rgb = cv2.resize(rgb, (self.w, self.h))
        x = np.expand_dims(rgb, axis=0)
        if self.inp["dtype"] == np.uint8:                      # fully quantized model
            x = x.astype(np.uint8)
        else:
            x = x.astype(np.float32)
            if PREPROCESS == "div255":
                x = x / 255.0
            elif PREPROCESS == "neg1_1":
                x = x / 127.5 - 1.0

        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)

        if self.out["dtype"] == np.uint8:                      # dequantize
            scale, zero = self.out["quantization"]
            y = (y - zero) * scale
        if y.min() < 0 or abs(float(y.sum()) - 1.0) > 0.01:    # not probabilities yet
            y = softmax(y)
        return y


# ----------------------------------------------------------------------
# APPLICATION
# ----------------------------------------------------------------------
class LeafApp:
    BG = "#f4f6f4"

    def __init__(self, root):
        self.root = root
        root.title("Leaf Disease Detection System")
        root.geometry("860x540")
        root.configure(bg=self.BG)
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
        root.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.fullscreen = False
        self.live = True
        self.current_frame = None     # latest frame from camera
        self.captured_frame = None    # frozen frame after Capture
        self.last_result = ""         # for file name when saving
        self.photo = None

        self.predictor = Predictor(MODEL_PATH, LABELS_PATH)
        self.build_ui()

        self.cap = open_camera(CAMERA_SOURCE)
        if not self.cap.isOpened():
            self.preview.config(text="Camera not found.\nCheck the webcam / phone URL\n(CAMERA_SOURCE). You can still\nuse the Upload button.",
                                fg="red", font=("Arial", 13))
            self.set_status("Camera error", "red")
            self.btn_capture.config(state="disabled")
        else:
            self.set_status("Camera ready", "#2e7d32")

        if self.predictor.demo:
            self.set_status(self.predictor.warning, "#e65100")
        elif self.predictor.warning:
            self.set_status(self.predictor.warning, "#e65100")

        self.update_preview()

    # ---------------- UI ----------------
    def build_ui(self):
        tk.Label(self.root, text="Leaf Disease Detection System",
                 font=("Arial", 18, "bold"), bg="#2e7d32", fg="white",
                 pady=8).pack(fill="x")

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # left: image
        left = tk.Frame(body, bg="black", width=PREVIEW_W, height=PREVIEW_H)
        left.pack(side="left")
        left.pack_propagate(False)
        self.preview = tk.Label(left, bg="black", fg="white")
        self.preview.pack(fill="both", expand=True)

        # right: results
        right = tk.Frame(body, bg=self.BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        tk.Label(right, text="Result", font=("Arial", 12, "bold"),
                 bg=self.BG, anchor="w").pack(fill="x")
        self.lbl_class = tk.Label(right, text="Point the camera at a leaf\nand press Capture",
                                  font=("Arial", 15, "bold"), bg=self.BG, fg="#333",
                                  wraplength=290, justify="left", anchor="w")
        self.lbl_class.pack(fill="x", pady=(4, 2))

        self.lbl_conf = tk.Label(right, text="Confidence: --", font=("Arial", 12),
                                 bg=self.BG, anchor="w")
        self.lbl_conf.pack(fill="x")
        self.bar = ttk.Progressbar(right, orient="horizontal", length=280,
                                   mode="determinate", maximum=100)
        self.bar.pack(fill="x", pady=4)

        self.lbl_msg = tk.Label(right, text="", font=("Arial", 10), bg=self.BG,
                                fg="#e65100", wraplength=290, justify="left", anchor="w")
        self.lbl_msg.pack(fill="x", pady=2)

        tk.Label(right, text="Top 3 predictions", font=("Arial", 11, "bold"),
                 bg=self.BG, anchor="w").pack(fill="x", pady=(10, 2))
        self.top_labels = []
        for _ in range(3):
            lb = tk.Label(right, text="", font=("Arial", 10), bg=self.BG,
                          anchor="w", wraplength=290, justify="left")
            lb.pack(fill="x")
            self.top_labels.append(lb)

        self.lbl_time = tk.Label(right, text="", font=("Arial", 9), bg=self.BG,
                                 fg="#666", anchor="w")
        self.lbl_time.pack(fill="x", pady=(8, 0))

        # buttons
        btns = tk.Frame(self.root, bg=self.BG)
        btns.pack(pady=4)
        style = {"font": ("Arial", 12, "bold"), "width": 9, "pady": 6}
        self.btn_capture = tk.Button(btns, text="Capture", bg="#2e7d32", fg="white",
                                     command=self.capture, **style)
        self.btn_upload = tk.Button(btns, text="Upload", bg="#ef6c00", fg="white",
                                    command=self.upload, **style)
        self.btn_retake = tk.Button(btns, text="Retake", bg="#1565c0", fg="white",
                                    command=self.retake, state="disabled", **style)
        self.btn_save = tk.Button(btns, text="Save", bg="#6a1b9a", fg="white",
                                  command=self.save, state="disabled", **style)
        self.btn_exit = tk.Button(btns, text="Exit", bg="#c62828", fg="white",
                                  command=self.exit_app, **style)
        for b in (self.btn_capture, self.btn_upload, self.btn_retake, self.btn_save, self.btn_exit):
            b.pack(side="left", padx=5)

        # status + disclaimer
        self.status = tk.Label(self.root, text="", font=("Arial", 10), bg=self.BG, anchor="w")
        self.status.pack(fill="x", padx=10)
        tk.Label(self.root,
                 text="Model-assisted indication only, not a confirmed diagnosis. "
                      "Consult an expert when unsure.",
                 font=("Arial", 9, "italic"), bg=self.BG, fg="#666").pack(pady=(0, 6))

    def set_status(self, text, color="#333"):
        self.status.config(text="Status: " + text, fg=color)

    # ---------------- Camera preview ----------------
    def show_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        img = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.preview.config(image=self.photo, text="")

    def update_preview(self):
        if self.live and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.current_frame = frame
                self.show_frame(frame)
            else:
                self.set_status("Lost camera signal", "red")
        self.root.after(30, self.update_preview)

    # ---------------- Actions ----------------
    def clear_results(self):
        self.lbl_class.config(text="Point the camera at a leaf\nand press Capture", fg="#333")
        self.lbl_conf.config(text="Confidence: --")
        self.bar["value"] = 0
        self.lbl_msg.config(text="")
        self.lbl_time.config(text="")
        for lb in self.top_labels:
            lb.config(text="")

    def capture(self):
        if self.current_frame is None:
            return
        self.analyse(self.current_frame.copy())

    def upload(self):
        path = filedialog.askopenfilename(
            title="Select a leaf photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is None:
            self.set_status("Could not read that image file", "red")
            return
        self.analyse(resize_max(frame, UPLOAD_MAX_SIDE))

    def analyse(self, frame):
        """Freeze the frame, run checks + model, show the result."""
        self.captured_frame = frame
        self.live = False
        self.show_frame(frame)
        self.btn_capture.config(state="disabled")
        self.btn_retake.config(state="normal")
        self.btn_save.config(state="disabled")
        self.clear_results()

        # 1) quality check
        ok, msg = check_quality(frame)
        if not ok:
            self.lbl_class.config(text="Cannot analyse image", fg="#e65100")
            self.lbl_msg.config(text=msg)
            self.set_status("Image rejected - please retake", "#e65100")
            self.last_result = ""
            return

        # 2) is there a leaf at all?
        leaf_ok, frac = check_leaf_present(frame)
        if not leaf_ok:
            self.lbl_class.config(text="No leaf detected", fg="#e65100")
            self.lbl_msg.config(text="This image does not look like a leaf "
                                     f"(plant colour {frac * 100:.0f}%). "
                                     "Place a single leaf close to the camera and retake.")
            self.set_status("No leaf found - please retake", "#e65100")
            self.last_result = "no_leaf"
            self.btn_save.config(state="normal")
            return

        # 3) prediction
        self.set_status("Analysing...", "#1565c0")
        self.root.update_idletasks()
        t0 = time.perf_counter()
        probs = self.predictor.predict(frame)
        dt = time.perf_counter() - t0

        order = np.argsort(probs)[::-1]
        best = int(order[0])
        conf = float(probs[best])
        name = self.predictor.labels[best] if best < len(self.predictor.labels) else f"Class {best}"

        # 4) top-3
        for i in range(min(3, len(order))):
            idx = int(order[i])
            lbl = self.predictor.labels[idx] if idx < len(self.predictor.labels) else f"Class {idx}"
            self.top_labels[i].config(text=f"{i + 1}. {pretty(lbl)}  -  {probs[idx] * 100:.1f}%")

        self.lbl_conf.config(text=f"Confidence: {conf * 100:.1f}%")
        self.bar["value"] = conf * 100
        self.lbl_time.config(text=f"Prediction time: {dt:.3f} s")

        # 5) decide what to show
        if name.strip().lower() in NOT_LEAF_NAMES:
            self.lbl_class.config(text="No leaf detected", fg="#e65100")
            self.lbl_msg.config(text="Place a single leaf in front of the camera and retake.")
            self.last_result = "no_leaf"
        elif conf < CONF_THRESHOLD:
            self.lbl_class.config(text="Not sure - please retake", fg="#f9a825")
            self.lbl_msg.config(text=f"Low confidence. Best guess: {pretty(name)}. "
                                     "Retake with better light or get expert confirmation.")
            self.last_result = "uncertain"
        else:
            healthy = "healthy" in name.lower()
            self.lbl_class.config(text=pretty(name), fg="#2e7d32" if healthy else "#c62828")
            self.last_result = name

        self.btn_save.config(state="normal")
        suffix = " (DEMO - fake result)" if self.predictor.demo else ""
        self.set_status("Done" + suffix, "#e65100" if self.predictor.demo else "#2e7d32")

    def retake(self):
        self.live = True
        self.captured_frame = None
        self.btn_capture.config(state="normal" if self.cap.isOpened() else "disabled")
        self.btn_retake.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.clear_results()
        self.preview.config(image="", text="" if self.cap.isOpened() else "No camera")
        self.set_status("Camera ready" if self.cap.isOpened() else "No camera - use Upload",
                        "#2e7d32" if self.cap.isOpened() else "#e65100")

    def save(self):
        if self.captured_frame is None:
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = (self.last_result or "rejected").replace(" ", "_").replace("/", "_")
        path = os.path.join(SAVE_DIR, f"{stamp}_{tag}.jpg")
        cv2.imwrite(path, self.captured_frame)
        self.set_status(f"Saved: {os.path.basename(path)}", "#6a1b9a")
        self.btn_save.config(state="disabled")

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def exit_app(self):
        try:
            if self.cap.isOpened():
                self.cap.release()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    LeafApp(root)
    root.mainloop()