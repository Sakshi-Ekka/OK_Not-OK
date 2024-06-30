import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)



# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, image = cap.read()                                 #ret is a boolean variable that returns true if frame is available.
        image = cv2.flip(image, 1)                              #flip by 180 degree (to avoid mirror image)
        image = cv2.resize(image, (800, 600))

        results = holistic.process(image)                        # Make Detections

        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Draw face landmarks
        mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Right hand
        mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
                                mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1))

        # Left Hand
        mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=1),
                                mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1))

        # Pose Detections
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
                                mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        cv2.imshow('Body Pose Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
