
import cv2
import mediapipe as mp
import os
import time

print("HOME GUARD AI - HD HAND TRACKING")

# Find the hand tracking model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("ERROR: hand_landmarker.task not found!")
    input("Press Enter to exit...")
    raise SystemExit

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

# Hand skeleton connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def identify_gesture(hand):
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    extended = [
        hand[tip].y < hand[joint].y
        for tip, joint in zip(tips, joints)
    ]

    if all(extended):
        return "OPEN HAND"

    if not any(extended):
        return "FIST"

    if extended[0] and not any(extended[1:]):
        return "POINTING"

    return "HAND DETECTED"


# Your original HD webcam setup
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("Webcam could not be opened!")
    input("Press Enter to exit...")
    raise SystemExit

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera resolution:", width, "x", height)
print("Hand tracking started!")
print("Press ESC to exit.")

last_timestamp = 0

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Camera frame could not be read.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = max(
                int(time.monotonic() * 1000),
                last_timestamp + 1
            )
            last_timestamp = timestamp

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            if result.hand_landmarks:

                for hand in result.hand_landmarks:

                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connecting lines
                    for start, end in CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            3
                        )

                    # Draw joint dots
                    for point in points:
                        cv2.circle(
                            frame,
                            point,
                            6,
                            (0, 0, 255),
                            -1
                        )

                    # Display gesture
                    gesture = identify_gesture(hand)

                    cv2.putText(
                        frame,
                        gesture,
                        (points[0][0], max(30, points[0][1] - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

            else:
                cv2.putText(
                    frame,
                    "SHOW YOUR HAND",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(
                "HOME GUARD AI - HD HAND TRACKING",
                frame
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed.")
