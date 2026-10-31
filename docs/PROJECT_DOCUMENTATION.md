# Project Documentation

## Project title
**Real-Time Object Detection and Tracking Using YOLO and DeepSORT**

## Objective
Build a computer-vision application that detects objects in a live camera stream or a video file and tracks detected objects across consecutive frames. The application displays bounding boxes, object labels, and tracking IDs.

## Tools and technologies

| Technology | Role |
|---|---|
| Python | Main programming language |
| OpenCV | Video capture, frame processing, drawing, and display |
| Ultralytics YOLO | Pre-trained object detection |
| DeepSORT | Multi-object tracking and track ID assignment |

## System workflow

```text
Webcam / Video file
        |
        v
OpenCV reads a frame
        |
        v
YOLO detects objects in the frame
        |
        v
Bounding boxes + confidence + class
        |
        v
DeepSORT associates detections across frames
        |
        v
Track ID + object class
        |
        v
OpenCV draws boxes and labels
        |
        v
Display frame / optionally save output
```

## Main modules

### 1. Input handling
`open_video_source()` accepts a webcam index such as `0` or a video file path. OpenCV's `VideoCapture` object reads frames until the stream ends or the user quits.

### 2. Detection
The YOLO model is loaded using `YOLO(args.model)`. For each frame, `model.predict()` returns detections. Each detection contains:
- `xyxy`: top-left and bottom-right box coordinates.
- `conf`: model confidence score.
- `cls`: predicted class index.

### 3. Tracking
The detections are converted into the format expected by DeepSORT: bounding box in left/top/width/height format, confidence, and class name. `update_tracks()` matches detections with existing tracks. Confirmed tracks expose a track ID and bounding-box coordinates.

### 4. Visualization
OpenCV's drawing functions render a rectangle and a label such as `person | ID: 3`. The ID helps distinguish multiple detected objects across frames.

### 5. Output
The processed frame is shown in a window. With the `--save` option, OpenCV's `VideoWriter` saves the annotated frames to an MP4 file.

## How to evaluate the result

Test with a short video containing multiple visible objects. Check:
1. Whether YOLO detects the objects of interest.
2. Whether bounding boxes reasonably follow the objects.
3. Whether the same visible object generally keeps the same ID across nearby frames.
4. Whether IDs remain understandable when objects cross or become temporarily hidden.
5. Whether the output video is saved and playable when `--save` is enabled.

Tracking quality is affected by object occlusion, similar-looking objects, fast movement, low resolution, and lighting. Track IDs are not guaranteed to remain unchanged after long occlusions or lost detections.

## Run commands

```bash
# Webcam
python main.py --source 0

# Video file
python main.py --source videos/sample.mp4

# Save annotated output
python main.py --source videos/sample.mp4 --save
```

## Expected output
A live video window showing detected objects enclosed in bounding boxes. Each confirmed track has a class label and a tracking ID. If saving is enabled, an annotated MP4 is created in the output directory.
