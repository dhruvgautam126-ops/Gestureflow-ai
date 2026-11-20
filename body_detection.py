
import cv2
import mediapipe as mp
import time
from pathlib import Path

# =====================================
# HOME GUARD AI - BODY DETECTION SYSTEM
# =====================================

# Model file location
MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Body landmark connections
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29),
    (27, 31), (24, 26), (26, 28),
    (28, 30), (28, 32)
]

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Unable to open webcam.")

print("HOME GUARD AI started!")
print("Press Q to exit.")

try:
    with PoseLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame unavailable.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Detect body landmarks
            timestamp = time.monotonic_ns() // 1_000_000

            result = detector.detect_for_video(
                mp_image, timestamp
            )

            height, width, _ = frame.shape

            if result.pose_landmarks:

                for person in result.pose_landmarks:

                    points = []

                    for landmark in person:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        points.append((x, y))

                    # Draw body connections
                    for start, end in CONNECTIONS:

                        if (
                            person[start].visibility > 0.5
                            and person[end].visibility > 0.5
                        ):
                            cv2.line(
                                frame,
                                points[start],
                                points[end],
                                (0, 255, 0),
                                3
                            )

                    # Draw landmarks
                    for index, landmark in enumerate(person):

                        if landmark.visibility > 0.5:
                            cv2.circle(
                                frame,
                                points[index],
                                5,
                                (0, 0, 255),
                                -1
                            )

                    # Draw person bounding box
                    visible_points = [
                        points[i]
                        for i, landmark in enumerate(person)
                        if landmark.visibility > 0.5
                    ]

                    if visible_points:
                        x_values = [p[0] for p in visible_points]
                        y_values = [p[1] for p in visible_points]

                        x1 = max(0, min(x_values))
                        y1 = max(0, min(y_values))
                        x2 = min(width - 1, max(x_values))
                        y2 = min(height - 1, max(y_values))

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            2
                        )

                # Human detected
                cv2.putText(
                    frame,
                    "HUMAN DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO HUMAN DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Display camera
            cv2.imshow(
                "HOME GUARD AI - BODY DETECTION",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("HOME GUARD AI stopped.")
