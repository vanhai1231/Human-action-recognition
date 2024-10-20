import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.layers import LSTM

# Định nghĩa lớp CustomLSTM để loại bỏ tham số 'time_major'
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Loại bỏ tham số 'time_major' nếu có
        super(CustomLSTM, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)  # Loại bỏ tham số 'time_major' từ config
        return super().from_config(config)

# Khởi tạo nhãn và các biến
label = "Khởi động..."
n_time_steps = 10  # Số khung hình để dự đoán
lm_list = []  # Danh sách để lưu các bộ điểm landmarks

# Khởi tạo MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Tải mô hình đã được huấn luyện với lớp CustomLSTM
model = tf.keras.models.load_model(
    "C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/model_3.h5",
    custom_objects={'CustomLSTM': CustomLSTM}
)

# Mở video từ tệp
video_path = "C:/Users/PC/Downloads/testchaybo2.mp4"  # Thay đổi đường dẫn đến tệp video của bạn
cap = cv2.VideoCapture(video_path)

# Hàm để chuyển đổi landmarks thành một danh sách thời gian
def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

# Hàm vẽ landmarks trên ảnh
def draw_landmark_on_image(mpDraw, results, img):
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h, w, c = img.shape
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img

# Hàm vẽ nhãn lớp hành động trên ảnh
def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

# Hàm dự đoán hành động
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)  
    results = model.predict(lm_list)

    action_index = np.argmax(results[0])  # Lấy chỉ số của hành động có xác suất cao nhất

    if action_index == 0:
        label = "Vay tay" 
    elif action_index == 1:
        label = "Lac nguoi" 
    elif action_index == 2:
        label = "Vo tay"  
    elif action_index == 3:
        label = "chay bo"  
    elif action_index == 4:
        label = "nhay"  
    else:
        label = "Khong xac dinh"  

    return label

# Số khung hình khởi động trước khi dự đoán
i = 0
warmup_frames = 60

# Kích thước khung hình (chiều rộng x chiều cao)
frame_width = 640
frame_height = 480


while True:
    success, img = cap.read()
    if not success:
        print("Không thể truy cập video.")
        break

    # Resize video cho dễ nhìn trong khung cố định
    img = cv2.resize(img, (frame_width, frame_height))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển ảnh sang RGB
    results = pose.process(imgRGB)  # Phân tích hình ảnh bằng MediaPipe Pose
    i += 1
    if i > warmup_frames:
        print("Bắt đầu dự đoán...")

        if results.pose_landmarks:
            # Tạo danh sách landmarks theo thời gian
            c_lm = make_landmark_timestep(results)
            lm_list.append(c_lm)

            # Khi đã đủ khung hình thì tiến hành dự đoán
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model, lm_list.copy(),))  # Sử dụng lm_list.copy() để tránh xung đột
                t1.start()
                lm_list = []  # Reset lm_list sau khi dự đoán

            # Vẽ landmarks trên ảnh
            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
