# Real-Time Object Detection and Tracking

A Python-based computer vision project that detects and tracks objects in real time using YOLO and DeepSORT. It processes webcam or video input, identifies objects, and assigns unique tracking IDs to follow them across frames.

## Features

- Real-time object detection using YOLO.
- Object tracking using DeepSORT.
- Supports webcam and video file input.
- Displays bounding boxes, class labels, and tracking IDs.
- Optional confidence scores.
- Option to save annotated video output.

## Tech Stack

- **Language:** Python
- **Object Detection:** YOLO (Ultralytics)
- **Object Tracking:** DeepSORT
- **Video Processing:** OpenCV

## Project Structure

```text
object-detection-and-tracking/
├── main.py
├── requirements.txt
├── videos/
├── output/
├── docs/
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/shaifalisahu13/object-detection-and-tracking.git
cd object-detection-and-tracking
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Run with webcam:**

```bash
python main.py
```

**Run with a video file:**

```bash
python main.py --source videos/my_video.mp4
```

**Save the annotated video:**

```bash
python main.py --source videos/my_video.mp4 --save
```

Press `Q` or `Esc` to stop the video window.

## How It Works

1. **Video Input:** OpenCV captures frames from a webcam or video file.
2. **Object Detection:** YOLO detects objects and identifies their classes and locations.
3. **Object Tracking:** DeepSORT associates detected objects across consecutive frames and assigns tracking IDs.
4. **Visualization:** OpenCV displays bounding boxes, class labels, and tracking IDs on the video frames.
5. **Output:** The processed video can optionally be saved.

## License

This project is licensed under the MIT License.
