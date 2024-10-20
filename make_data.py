import cv2
import mediapipe as mp
import pandas as pd

# Đọc video từ file thay vì từ webcam
video_path = 'C:/Users/PC/Downloads/nhay3.mp4'
cap = cv2.VideoCapture(video_path)

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "nhay"

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        # Thay đổi kích thước khung hình (tùy chỉnh theo độ phân giải mong muốn)
        frame_resized = cv2.resize(frame, (640, 480))  # Ví dụ: resize về 640x480

        # Hiển thị video đã được thay đổi kích thước
        cv2.imshow("image", frame_resized)
        
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Write vào file csv
df = pd.DataFrame(lm_list)
output_path = "C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/JUMP/" + label + "3" + ".txt"
df.to_csv(output_path)
cap.release()
cv2.destroyAllWindows()
