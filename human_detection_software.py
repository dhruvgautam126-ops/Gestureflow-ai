
import cv2
import mediapipe as mp
import os
import time

print("====================================")
print(" HOME GUARD AI - HUMAN TRACKING")
print("====================================")

# --------------------------------------------------
# 1. MODEL FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSE_MODEL = os.path.join(
    BASE_DIR, "pose_landmarker_lite.task"
)

HAND_MODEL = os.path.join(
    BASE_DIR, "hand_landmarker.task"
)

if not os.path.exists(POSE_MODEL):
    raise FileNotFoundError(
        "pose_landmarker_lite.task not found!"
    )

if not os.path.exists(HAND_MODEL):
    raise FileNotFoundError(
        "hand_landmarker.task not found!"
    )

# --------------------------------------------------
# 2. MEDIAPIPE SETUP
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandOptions = mp.tasks.vision.HandLandmarkerOptions

pose_options = PoseOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_options = HandOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# 3. BODY CONNECTIONS
# --------------------------------------------------

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32)
]

# --------------------------------------------------
# 4. HAND CONNECTIONS
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# --------------------------------------------------
# 5. HAND GESTURE RECOGNITION
# --------------------------------------------------

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


# --------------------------------------------------
# 6. DRAW BODY TRACKING
# --------------------------------------------------

def draw_body(frame, pose_result, width, height):

    person_count = 0

    if pose_result.pose_landmarks:

        for landmarks in pose_result.pose_landmarks:

            person_count += 1
            points = []

            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            # Draw body connections
            for start, end in POSE_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            # Draw body landmarks
            for point in points:

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1
                )

            # Draw bounding box
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]

            x_min = max(0, min(x_values))
            y_min = max(0, min(y_values))
            x_max = min(width - 1, max(x_values))
            y_max = min(height - 1, max(y_values))

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"PERSON {person_count}",
                (x_min, max(25, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    return person_count


# --------------------------------------------------
# 7. DRAW HAND TRACKING
# --------------------------------------------------

def draw_hands(frame, hand_result, width, height):

    hand_count = 0

    if hand_result.hand_landmarks:

        for hand in hand_result.hand_landmarks:

            hand_count += 1
            points = []

            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

            # Draw hand connections
            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    3
                )

            # Draw hand landmarks
            for point in points:

                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )

            # Recognize gesture
            gesture = identify_gesture(hand)

            cv2.putText(
                frame,
                gesture,
                (points[0][0], max(30, points[0][1] - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    return hand_count


# --------------------------------------------------
# 8. WEBCAM SETUP
# --------------------------------------------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():

    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():

    raise RuntimeError("Webcam could not be opened!")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera resolution:", width, "x", height)
print("Body and hand tracking started!")
print("Press ESC to exit.")

# --------------------------------------------------
# 9. MAIN APPLICATION
# --------------------------------------------------

last_timestamp = 0

try:

    with PoseLandmarker.create_from_options(
        pose_options
    ) as pose_landmarker, HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame could not be read.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Create increasing timestamp
            timestamp = max(
                int(time.monotonic() * 1000),
                last_timestamp + 1
            )

            last_timestamp = timestamp

            # Run body detection
            pose_result = pose_landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # Run hand detection
            hand_result = hand_landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            # Draw body tracking
            people = draw_body(
                frame,
                pose_result,
                width,
                height
            )

            # Draw hand tracking
            hands = draw_hands(
                frame,
                hand_result,
                width,
                height
            )

            # Display information
            cv2.putText(
                frame,
                f"PEOPLE: {people}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"HANDS: {hands}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if people > 0:
                status = "HUMAN DETECTED"
                status_color = (0, 255, 0)
            else:
                status = "NO HUMAN DETECTED"
                status_color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            # Show combined application
            cv2.imshow(
                "HOME GUARD AI - HUMAN TRACKING SOFTWARE",
                frame
            )

            # Exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")
    print("HOME GUARD AI stopped.")
