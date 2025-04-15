import sys
import cv2
import cv2.data
import numpy as np
from keras.api.models import load_model
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.dont_write_bytecode = True


class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel()
        self.label.setFixedSize(800, 600)

        self.upload_button = QPushButton("Upload Photo")
        self.upload_button.clicked.connect(self.upload_photo)

        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Tải mô hình
        self.model = load_model("model.keras")

    def upload_photo(self):
        # Mở hộp thoại thư mục để chọn ảnh
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if file_name:
            self.detect_faces(file_name)

    def detect_faces(self, image_path):
        # Tải thuật toán cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
        )

        # Kiểm tra xem classifier được tải hay không
        if face_cascade.empty():
            print("Error: Classifier not loaded")
            return

        # Đọc ảnh đầu vào
        img = cv2.imread(image_path)

        # Biến ảnh xám thành ảnh màu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        # Vẽ hình chữ nhật xung quanh khuôn mặt và dự đoán giới tính và tuổi
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Trích xuất vùng khuôn mặt
            face = img[y : y + h, x : x + w]

            # Thay đổi kích thước khuôn mặt để phù hợp với đầu vào của mô hình
            face_resized = cv2.resize(face, (128, 128))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # Dự đoán giới tính và tuổi
            gender_pred, age_pred = self.model.predict(face_resized)
            gender = "Male" if gender_pred[0] < 0.5 else "Female"
            age = int(age_pred[0])

            # Chú thích ảnh với giới tính và tuổi
            label = f"{gender}, {age}"
            cv2.putText(
                img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
            )

        # Thay đổi kích thước ảnh để phù hợp với QLabel
        height, width, _ = img.shape
        max_height, max_width = self.label.height(), self.label.width()
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # Chuyển đổi ảnh sang định dạng Qt (giữ nguyên màu gốc)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qt_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Hiển thị ảnh trong QLabel
        self.label.setPixmap(QPixmap.fromImage(qt_img.rgbSwapped()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec())
