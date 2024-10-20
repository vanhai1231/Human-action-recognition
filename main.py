import sys

from deepsort.detection import Detection

sys.path.insert(0, 'C:/Users/PC/Downloads/Object Tracking')
import collections
import numpy as np
import cv2
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO
from deepsort import DeepSortTracker  # Import DeepSORT tracker

# Định nghĩa lớp CustomLSTM để xử lý tham số time_major nếu có
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Xóa bỏ tham số time_major nếu có
        super(CustomLSTM, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)  # Đảm bảo rằng nó cũng được loại bỏ từ config
        return super().from_config(config)

# Hàm để vẽ khung xương nửa người trở lên
def draw_upper_body_landmarks(image, landmarks):
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            if idx in {11, 12, 13, 14, 15, 16, 23, 24}:  # Các chỉ số cho khung xương nửa trên
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Vẽ điểm với màu xanh lá cây

# Hàm để trích xuất keypoints từ kết quả của Mediapipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose

# Luồng riêng để xử lý webcam hoặc video
class VideoThread(QThread):
    update_frame = pyqtSignal(QImage)

    def __init__(self, model, actions, video_source=0, threshold=0.8):
        super(VideoThread, self).__init__()
        self.model = model
        self.actions = actions
        self.threshold = threshold
        self.sequence = collections.deque(maxlen=30)
        self.video_source = video_source
        self.cap = None
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.running = True

        # Khởi tạo YOLOv8 và DeepSORT
        # self.yolo_model = YOLO('C:\\Users\\PC\\Downloads\\Object Tracking\\Object tracking\\yolov8n.pt')  # Đường dẫn mô hình YOLOv8
        # self.tracker = DeepSortTracker()

    def run(self):
        # Mở nguồn video (webcam hoặc file video)
        self.cap = cv2.VideoCapture(self.video_source)
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Phát hiện vật thể bằng YOLOv8
            # yolo_results = self.yolo_model(frame)
            # bboxes = []
            # for r in yolo_results:
            #     for box in r.boxes:
            #         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            #         conf = float(box.conf[0])
            #         print(f"YOLO Detection - BBox: {x1, y1, x2, y2}, Confidence: {conf}")
            #         if conf > 0.5:  # Chỉ thêm các bounding box với độ tin cậy cao
            #             bboxes.append([x1, y1, x2, y2])

            # Vẽ bounding box từ YOLOv8 lên khung hình
            # for bbox in bboxes:
            #     x1, y1, x2, y2 = bbox
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Màu xanh lam cho bounding box

            # Theo dõi vật thể bằng DeepSORT
            # detections = [Detection(bbox, 1.0, None) for bbox in bboxes]
            # print("Detections:", detections)

            # Cập nhật theo dõi với frame và detections
            # tracks = self.tracker.update(detections)
            # print(f"Tracks: {tracks}")  # Kiểm tra xem giá trị của tracks là gì

            # if tracks is None:
            #     print("Không có đối tượng nào để theo dõi")
            #     tracks = []

            # Vẽ bounding boxes và ID của đối tượng được theo dõi từ DeepSORT
            # for track in tracks:
            #     if not track.is_confirmed() or track.time_since_update > 1:
            #         print(f"Track {track.track_id} không được xác nhận hoặc quá thời gian cập nhật.")
            #         continue
            #     bbox = track.to_tlbr()
            #     track_id = track.track_id
            #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)  # Màu xanh lá cây cho bounding box DeepSORT
            #     cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Xử lý Mediapipe Pose cho nhận diện hành động
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            # Vẽ khung xương nửa người trở lên
            draw_upper_body_landmarks(image_rgb, results.pose_landmarks)

            # Trích xuất keypoints và thêm vào hàng đợi
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)

            # Nhận diện hành động khi có đủ 30 khung hình
            if len(self.sequence) == 30:
                input_data = np.expand_dims(self.sequence, axis=0)
                prediction = self.model.predict(input_data)[0]
                if np.max(prediction) > self.threshold:
                    predicted_action = self.actions[np.argmax(prediction)]
                    # Hiển thị nhãn hành động lên khung hình
                    cv2.putText(image_rgb, f'Action: {predicted_action}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Chuyển đổi khung hình thành QImage để hiển thị
            image = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888)
            self.update_frame.emit(image)

    def stop(self):
        self.running = False
        self.cap.release()


# Class giao diện chính của ứng dụng
class Ui_Dialog(QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()

        self.setWindowTitle("Action Recognition and Object Tracking")

        # Layout chính của ứng dụng
        layout = QVBoxLayout()

        # Nhãn hiển thị video từ webcam hoặc video file
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        # Nút bật webcam
        self.webcam_button = QPushButton("Start Webcam")
        self.webcam_button.clicked.connect(self.start_webcam)
        layout.addWidget(self.webcam_button)

        # Nút chọn video
        self.video_button = QPushButton("Open Video")
        self.video_button.clicked.connect(self.open_video)
        layout.addWidget(self.video_button)

        self.setLayout(layout)

        # Đường dẫn tới mô hình đã huấn luyện
        model_path = r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/model1.h5"

        # Tải mô hình
        if os.path.exists(model_path):
            self.model = load_model(model_path, custom_objects={'LSTM': CustomLSTM})
        else:
            print("Model file not found!")
        #"RUNNING", "WALKING", "STANDING", "FALL", "HANDCLAPPING"
        # Danh sách các nhãn hành động
        self.actions = np.array(["Vỗ tay"])

        # Luồng webcam hoặc video
        self.video_thread = None

    def start_webcam(self):
        # Bắt đầu luồng video
        if self.video_thread is None or not self.video_thread.isRunning():
            self.video_thread = VideoThread(self.model, self.actions)
            self.video_thread.update_frame.connect(self.update_video_frame)
            self.video_thread.start()

    def open_video(self):
        # Mở file video
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            if self.video_thread is None or not self.video_thread.isRunning():
                self.video_thread = VideoThread(self.model, self.actions, video_source=file_name)
                self.video_thread.update_frame.connect(self.update_video_frame)
                self.video_thread.start()

    def update_video_frame(self, image):
        # Cập nhật nhãn video
        scaled_image = image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def closeEvent(self, event):
        # Dừng luồng khi đóng ứng dụng
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_Dialog()
    window.show()
    sys.exit(app.exec_())
