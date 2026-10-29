# IoT-Based Tomato Leaf Disease Detection (No Backend)

Raspberry Pi + USB webcam + MobileNetV3-Large (TFLite) + Tkinter interface.
Everything runs locally on the Pi: no server, no database, no internet needed at run time.

> Results are a **model-assisted indication**, not a confirmed diagnosis.

## Project structure

```
leaf_disease_detection/
├── README.md                      <- this file
├── requirements.txt               <- runtime packages
├── setup_pi.sh                    <- one-time Raspberry Pi setup
├── .gitignore
│
├── data_collection/               <- run on Pi/PC to capture your own images
│   └── collect_images.py
│       (creates webcam_images/<class>/*.jpg)
│
├── training/                      <- run on PC / Google Colab
│   ├── prepare_dataset.py         <- builds dataset/train|val|test
│   ├── train_model.ipynb          <- trains + exports model.tflite & labels.txt
│   └── requirements_training.txt  <- only for local Jupyter
│
└── raspberry_pi_app/              <- the final application
    ├── app.py                     <- Tkinter frontend (camera + model + UI)
    ├── test_camera.py             <- webcam check
    ├── model.tflite               <- FROM COLAB (you add this)
    ├── labels.txt                 <- FROM COLAB (you add this)
    ├── labels_example.txt         <- format example only
    └── captures/                  <- images saved with the Save button
```

## Classes (tomato)

Not_a_leaf, Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Healthy, Tomato_Late_blight,
Tomato_Leaf_Mold, Tomato_Mosaic_virus, Tomato_Septoria_leaf_spot, Tomato_Spider_mites,
Tomato_Target_Spot, Tomato_Yellow_Leaf_Curl_Virus

You can use a subset (pass `--classes` to `prepare_dataset.py` and edit `CLASSES` in `collect_images.py`).

## Workflow

### 1. Set up the Raspberry Pi
```bash
bash setup_pi.sh
source venv/bin/activate
cd raspberry_pi_app
python3 test_camera.py        # check webcam, brightness and sharpness values
python3 app.py                # works now in DEMO MODE (fake predictions) to test the UI
```

### 2. Get PlantVillage tomato images
Download the PlantVillage dataset (e.g. from Kaggle) to your PC. You only need the `Tomato___*` folders.

### 3. Collect your own webcam images (on the Pi, in your final setup)
```bash
cd data_collection
python3 collect_images.py
```
`N`/`P` change class, `SPACE` saves, `A` toggles auto mode, `Q` quits.
Collect **50-100+ images per class** (many different leaves), plus plenty of `Not_a_leaf`.
Copy `webcam_images/` to your PC when done.

### 4. Build the dataset (on your PC)
```bash
cd training
python prepare_dataset.py --pv "PATH/TO/PlantVillage" --real "../data_collection/webcam_images" --out dataset
```
Zip the `dataset` folder as `leaf_dataset.zip` (the zip's top-level folder must be `dataset`) and upload it to Google Drive.

### 5. Train in Google Colab
1. Open https://colab.research.google.com -> File -> Upload notebook -> `train_model.ipynb`.
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Check `ZIP_PATH` in the settings cell, then Runtime -> Run all.
4. Read the classification report, confusion matrix and the threshold table.
5. Download `model.tflite` and `labels.txt` (the last cell does this).

### 6. Deploy
Copy `model.tflite` and `labels.txt` into `raspberry_pi_app/`, then:
```bash
cd raspberry_pi_app
python3 app.py
```
The DEMO banner disappears and predictions are real.

### 7. Tune and re-test
* Set `CONF_THRESHOLD` in `app.py` from the notebook's threshold table.
* Test with real leaves in different light / backgrounds.
* Use the **Save** button to keep wrong predictions, add them to `webcam_images/<correct_class>/`, rebuild the dataset and retrain.

## Important rules (common bugs)
* **No `/255` anywhere.** Keras MobileNetV3 has preprocessing built in and expects 0-255 pixels. `app.py` already feeds raw values.
* `labels.txt` must come from the notebook (same class order as training).
* OpenCV frames are BGR; `app.py` converts to RGB before inference.
* Keep the camera setup (distance 15-25 cm, plain light background, steady light) the same when collecting data and when running the app.
* Judge the model on **real webcam test images**, not only on PlantVillage test images.

## Optional: auto-start on boot (Raspberry Pi OS desktop)
Create `~/.config/autostart/leaf.desktop`:
```
[Desktop Entry]
Type=Application
Name=Leaf Detector
Exec=/home/pi/leaf_disease_detection/venv/bin/python /home/pi/leaf_disease_detection/raspberry_pi_app/app.py
```
(Adjust the paths and username.)

## Troubleshooting
| Problem | Fix |
|---|---|
| "Camera not found" | Try `CAMERA_INDEX = 1`; check `ls /dev/video*` |
| DEMO MODE banner | `model.tflite` missing, or no TFLite runtime installed |
| "labels.txt has N names but model outputs M" | Wrong labels file; use the one from the notebook |
| Good images rejected | Lower `BLUR_THRESHOLD` / adjust brightness limits in `app.py` |
| Always predicts one class | Class imbalance or too little real data; collect more, retrain |
| High test accuracy, poor real results | Training data too clean; add more webcam images |
| `tflite-runtime` won't install | `pip install ai-edge-litert` |
