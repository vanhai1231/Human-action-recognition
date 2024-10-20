import cv2
import numpy as np
import joblib

# Tải mô hình đã lưu
decision_tree = joblib.load('decision_tree_model.pkl')

# Hàm trích xuất đặc trưng từ frame
def extract_features(frame):
    # Thay đổi kích thước frame để đồng nhất
    resized_frame = cv2.resize(frame, (100, 100))  # Thay đổi kích thước về 100x100
    # Chuyển đổi màu sắc từ BGR sang RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Tính toán các đặc trưng (ví dụ: màu sắc trung bình)
    mean_color = np.mean(rgb_frame, axis=(0, 1))  # Màu sắc trung bình (3 đặc trưng)
    
    # Trả về các đặc trưng như là vector 1D
    return mean_color  # Trả về màu sắc trung bình như là đặc trưng (3 giá trị)

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tiền xử lý frame
        feature = extract_features(frame)  # Gọi hàm trích xuất đặc trưng từ frame
        features.append(feature)

    cap.release()
    return np.array(features)

# Đường dẫn đến video
video_path = 'C:/Users/PC/Downloads/testchaybo2.mp4'
video_features = extract_features_from_video(video_path)

# Kiểm tra số lượng đặc trưng
print("Shape of raw video features:", video_features.shape)  # In ra kích thước của đặc trưng video

# Nếu số lượng frame > 0, tính toán các đặc trưng cuối cùng
if video_features.shape[0] > 0:
    # Tính trung bình của các đặc trưng từ tất cả các frame
    video_features_mean = np.mean(video_features, axis=0).reshape(1, -1)  # Kết quả sẽ có shape (1, 3)
else:
    video_features_mean = np.zeros((1, 3))  # Trả về vector không nếu không có frame

# Đảm bảo số lượng đặc trưng khớp với yêu cầu của mô hình
# Nếu mô hình yêu cầu 100 đặc trưng, bạn sẽ cần thêm các đặc trưng khác hoặc điều chỉnh
if video_features_mean.shape[1] < 100:
   
    video_features_combined = np.pad(video_features_mean, ((0, 0), (0, 100 - video_features_mean.shape[1])), mode='constant')
else:
    video_features_combined = video_features_mean  # Giữ nguyên nếu đã đủ 100

print("Shape of combined video features:", video_features_combined.shape)  # In ra kích thước của đặc trưng video

# Dự đoán trên đặc trưng video
y_pred_video = decision_tree.predict(video_features_combined)

# In kết quả dự đoán
print("Predicted label for video:", y_pred_video)
