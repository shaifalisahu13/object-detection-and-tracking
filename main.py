"""
Real-time object detection and tracking using YOLO + DeepSORT.

Examples:
    python main.py --source 0
    python main.py --source videos/sample.mp4
    python main.py --source 0 --save
    python main.py --source videos/sample.mp4 --model yolov8s.pt --confidence 0.4
"""

import argparse
import sys
from pathlib import Path

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect objects with YOLO and track them with DeepSORT."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0, 1, ...) or path to a video file. Default: 0",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model weights. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence from 0 to 1. Default: 0.35",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated video to the output folder.",
    )
    parser.add_argument(
        "--output",
        default="output/tracked_output.mp4",
        help="Output video path when --save is used.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames to keep a track alive without a detection. Default: 30",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Include detection confidence in labels.",
    )
    return parser.parse_args()


def open_video_source(source_text):
    """Open a webcam index or a video file."""
    if source_text.isdigit():
        source = int(source_text)
    else:
        source = source_text

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source_text}. "
            "Check the webcam connection or video file path."
        )
    return capture


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    print(f"Loading YOLO model: {args.model}")
    # On the first run, Ultralytics downloads the selected weights automatically.
    model = YOLO(args.model)

    tracker = DeepSort(
        max_age=args.max_age,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=100,
    )

    capture = open_video_source(args.source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0

    writer = None
    window_name = "YOLO + DeepSORT | Press Q to quit"

    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Video stream ended or a frame could not be read.")
                break

            # YOLO detects objects in the current frame.
            results = model.predict(
                source=frame,
                conf=args.confidence,
                verbose=False,
            )

            detections_for_tracker = []
            names = model.names

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = names[class_id]

                    width = max(0.0, x2 - x1)
                    height = max(0.0, y2 - y1)
                    if width == 0 or height == 0:
                        continue

                    # DeepSORT expects [left, top, width, height], confidence, class.
                    detections_for_tracker.append(
                        ([x1, y1, width, height], confidence, class_name)
                    )

            # DeepSORT associates detections across frames and assigns track IDs.
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                left, top, right, bottom = map(int, track.to_ltrb())
                track_id = track.track_id
                class_name = str(track.get_det_class() or "object")

                cv2.rectangle(
                    frame, (left, top), (right, bottom), (40, 220, 40), 2
                )

                label = f"{class_name} | ID: {track_id}"
                if args.show_confidence and track.get_det_conf() is not None:
                    label += f" | {track.get_det_conf():.2f}"

                # Keep the label background inside the frame.
                text_y = max(22, top - 8)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                cv2.rectangle(
                    frame,
                    (left, text_y - text_height - baseline),
                    (left + text_width, text_y + baseline),
                    (40, 220, 40),
                    thickness=-1,
                )
                cv2.putText(
                    frame,
                    label,
                    (left, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "YOLO detection + DeepSORT tracking",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save:
                if writer is None:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_path), fourcc, fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output: {output_path}")
                    print(f"Saving annotated video to: {output_path}")
                writer.write(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)
