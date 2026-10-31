# Real-Time Object Detection and Tracking (CodeAlpha Task 2)

A Python project that detects objects in webcam/video input with **YOLO** and follows detected objects across frames with **DeepSORT**. Each confirmed track is displayed with a bounding box, class label, and tracking ID.

## Features

- Live webcam input or a video file as input.
- Pre-trained YOLO model for object detection.
- DeepSORT tracking to associate detections between frames.
- Bounding boxes, object class labels, and persistent track IDs.
- Optional confidence values and annotated-video export.
- Command-line options for model weights and detection threshold.

## 1. Requirements

- Windows 10/11, Linux, or macOS.
- Python 3.9–3.12 recommended.
- A webcam for live input, or a video file.
- Internet access on the first run to download the YOLO weights.
- CPU works; a compatible NVIDIA GPU can make inference faster.

> Package compatibility can vary by Python/PyTorch release. If installation fails, use a fresh Python 3.10 or 3.11 virtual environment.

## 2. Project structure

```text
object_detection_tracking_task2/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── videos/
│   └── README.txt
├── output/
│   └── README.txt
└── docs/
    └── PROJECT_DOCUMENTATION.md
```

The `videos` folder is for your own test clips. The `output` folder is where annotated videos are saved. Empty folders are retained using README files.

## 3. Setup on Windows

Open Command Prompt or PowerShell in the extracted project folder.

### Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

If `py -3.11` is unavailable, use `python -m venv .venv` with an installed supported Python version.

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch is installed as a dependency of Ultralytics. If you specifically need CUDA acceleration, install the PyTorch build that matches your CUDA setup using the official PyTorch installation selector.

## 4. Run the project

### Use the default webcam

```powershell
python main.py
```

The default source is webcam index `0`.

### Use a video file

Place a video in the `videos` folder, then run:

```powershell
python main.py --source videos/my_video.mp4
```

You can also pass an absolute path:

```powershell
python main.py --source "C:\Users\YourName\Videos\traffic.mp4"
```

### Save the annotated result

```powershell
python main.py --source videos/my_video.mp4 --save
```

The output is written to `output/tracked_output.mp4`.

### Useful options

```powershell
python main.py --source 0 --confidence 0.45
python main.py --source 0 --model yolov8s.pt
python main.py --source videos/my_video.mp4 --save --output output/result.mp4
python main.py --source 0 --show-confidence
python main.py --source 0 --max-age 45
```

| Option | Purpose | Default |
|---|---|---|
| `--source` | Webcam index or video path | `0` |
| `--model` | YOLO weights file/name | `yolov8n.pt` |
| `--confidence` | Minimum detection confidence | `0.35` |
| `--save` | Save annotated video | Off |
| `--output` | Saved video destination | `output/tracked_output.mp4` |
| `--max-age` | Frames a track can remain unmatched | `30` |
| `--show-confidence` | Add confidence to the displayed label | Off |

Press **Q** or **Esc** in the video window to stop.

## 5. How it works

1. **Video capture:** OpenCV (`cv2.VideoCapture`) reads frames from a webcam or video file.
2. **Object detection:** YOLO processes each frame and returns bounding-box coordinates, class IDs, and confidence scores.
3. **Detection formatting:** Each detection is converted to `[left, top, width, height]`, confidence, and class label for DeepSORT.
4. **Object tracking:** DeepSORT uses motion information and appearance features to associate current detections with existing tracks. It assigns a track ID to each tracked object.
5. **Visualization:** OpenCV draws each confirmed track's bounding box, class label, and ID on the frame.
6. **Display/export:** OpenCV displays the annotated frame and optionally writes it to an MP4 file.

## 6. Troubleshooting

- **Camera does not open:** Close apps that may already be using the webcam. Try `--source 1` if your camera uses another device index.
- **Video path error:** Check the spelling and use quotes around paths containing spaces.
- **`ModuleNotFoundError`:** Activate `.venv` and run `pip install -r requirements.txt` again.
- **First run takes time:** Ultralytics downloads `yolov8n.pt` automatically if the weights are not already available.
- **Slow performance:** Use `yolov8n.pt`, reduce the input video resolution, or use a supported GPU.
- **IDs change after a long disappearance:** This is normal for tracking-by-detection. DeepSORT may create a new track when an object is lost for too long.
- **Window does not appear in a notebook/headless server:** Run the script from a desktop terminal with GUI support. `cv2.imshow` needs a display.

## 7. GitHub upload

Create a new GitHub repository, for example `object-detection-and-tracking`.

From the project directory:

```bash
git init
git add .
git commit -m "Add YOLO and DeepSORT object detection tracking"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/object-detection-and-tracking.git
git push -u origin main
```

Do not upload `.venv/`, large model weight files, or generated output videos unless your instructor specifically requests them. The `.gitignore` excludes these by default.

## 8. Important notes

- The default `yolov8n.pt` model is a small pre-trained COCO model. Its available classes are limited to the classes it was trained to recognize.
- This is a tracking-by-detection pipeline: YOLO detects objects independently in each frame, and DeepSORT associates those detections over time.
- A tracking ID is an identifier within the current run, not a real-world identity. IDs may change if an object disappears or tracking is interrupted.
- Results depend on lighting, camera quality, object size, occlusion, motion blur, and model confidence threshold.

## 9. Task checklist

- [x] Set up real-time video input using OpenCV.
- [x] Use a pre-trained YOLO model for object detection.
- [x] Process frames and draw bounding boxes.
- [x] Apply object tracking using DeepSORT.
- [x] Display class labels and tracking IDs in real time.

## License

This project is provided for educational use. See `LICENSE` for the MIT License text.
