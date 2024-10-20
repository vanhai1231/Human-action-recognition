import os
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

DATA_PATH = os.path.join("DATA")
actions = ["RUNNING", "WALKING", "STANDING", "FALL", "HANDCLAPPING"]
video_to_action = {
    "chay.mp4": "RUNNING",
    "di.mp4": "WALKING",
    "dung.mp4": "STANDING",
    "nga.mp4": "FALL",
    "votay.mp4": "HANDCLAPPING"
}
no_sequences = 1775
sequence_length = 30

# Tạo các thư mục tương ứng
for action in actions:
    for sequence in range(no_sequences):
        path = os.path.join(DATA_PATH, action, str(sequence))
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created")
        else:
            print(f"Directory {path} already exists")

# Địa chỉ chứa các video
video_path = "C:\\Users\\PHUCTOAN\\PycharmProjects\\video"
video_list = os.listdir(video_path)

def extract_keypoints(results):
    keypoints = []
    for res in [results.pose_landmarks, results.face_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
        if res is not None:
            for landmark in res.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            keypoints.extend([0] * 21 * 4)  # Chọn số lượng landmark
    return np.array(keypoints)

# Sử dụng tool MediaPipe Holistic
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pTime = datetime.now().timestamp()

# Xử lý mỗi video
for video in video_list:
    if video not in video_to_action:
        continue  # Bỏ qua nếu video không nằm trong danh sách đã định

    action = video_to_action[video]
    video_file = os.path.join(video_path, video)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Cannot open video file {video_file}")
        continue

    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ref, frame = cap.read()
            if not ref:
                print(f"Failed to read frame at sequence {sequence}, frame {frame_num} for video {video}")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

            cTime = datetime.now().timestamp()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(image, f"FPS: {int(fps)}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 190), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            np.save(npy_path, keypoints)
            print(f"Keypoints for {action}, sequence {sequence}, frame {frame_num} saved at {npy_path}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
cv2.destroyAllWindows()
mp_holistic.close()
