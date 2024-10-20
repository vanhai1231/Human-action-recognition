import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
import joblib

# Khởi tạo danh sách X và y
X = []
y = []
no_of_timesteps = 10

# Hàm để đọc dữ liệu từ các tệp CSV
def add_data_from_csv(file_path, label):
    dataset = pd.read_csv(file_path).iloc[:, 1:].values  # Bỏ cột đầu tiên nếu không chứa dữ liệu hữu ích
    n_sample = len(dataset)
    for i in range(no_of_timesteps, n_sample):
        X.append(dataset[i-no_of_timesteps:i, :])
        y.append(label)

# Thêm dữ liệu các hành động đã có
add_data_from_csv(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/lac nguoi.txt", 1)  # Lắc người
add_data_from_csv(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/Vay tay.txt", 0)  # Đưa tay
add_data_from_csv(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/Vo tay.txt", 2)  # Vỗ tay

# Đọc nhiều file chạy bộ
running_files = glob.glob(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/RUNNING/chay bo*.txt")
for running_file in running_files:
    add_data_from_csv(running_file, 3)  # Nhãn cho hành động "chạy bộ"

jump_files = glob.glob(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/DATA/JUMP/nhay*.txt")
for jump_file in jump_files:
    add_data_from_csv(jump_file, 4)

# Chuyển đổi X và y thành numpy array
X, y = np.array(X), np.array(y)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Chuyển đổi nhãn thành dạng one-hot encoding
y = to_categorical(y, num_classes=5)

# Chia tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thiết lập mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu', name="feature_layer"))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation="softmax"))

# Biên dịch mô hình
model.compile(optimizer=Adam(), metrics=['accuracy'], loss='categorical_crossentropy')
# Sau khi chia tập dữ liệu
np.save(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/y_test.npy", y_test)

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test))

# Lưu lại mô hình
model.save(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/model_6.h5")

# Trích xuất đặc trưng
model_feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer("feature_layer").output)
X_train_features = model_feature_extractor.predict(X_train)
X_test_features = model_feature_extractor.predict(X_test)

print("Shape of X_train_features:", X_train_features.shape)
print("Shape of X_test_features:", X_test_features.shape)

# Lưu đặc trưng vào file
np.save(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/X_train_features.npy", X_train_features)
np.save(r"C:/Users/PC/Downloads/human_activity_recognition-/human_activity_recognition-/weight/X_test_features.npy", X_test_features)

# Huấn luyện Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_features, np.argmax(y_train, axis=1))  # Chuyển đổi y_train về dạng nhãn số

# Dự đoán và đánh giá
y_pred = decision_tree.predict(X_test_features)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print(f"Accuracy of Decision Tree: {accuracy:.2f}")

joblib.dump(decision_tree, 'decision_tree_model.pkl')