import torch
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Tải mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process_image(img_path):
    # Đọc ảnh
    img = cv2.imread(img_path)

    # Phát hiện đối tượng
    results = model(img)

    # Hiển thị kết quả phát hiện
    results.show()

    # Trích xuất thông tin đối tượng phát hiện được
    detections = results.pandas().xyxy[0]  # Lấy kết quả dưới dạng DataFrame

    # Lọc các đối tượng là động vật (nếu YOLOv5 nhận dạng được)
    animals = detections[detections['name'].isin(['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])]

    # In thông tin của các loài động vật được phát hiện
    for idx, row in animals.iterrows():
        print(f"Đối tượng: {row['name']}, Xác suất: {row['confidence']:.2f}, Vị trí: [{row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}]")

# Vòng lặp chọn ảnh
while True:
    # Sử dụng tkinter để chọn ảnh
    Tk().withdraw()  # Ẩn cửa sổ chính của tkinter
    img_path = askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

    # Kiểm tra nếu người dùng đóng cửa sổ mà không chọn ảnh nào
    if not img_path:
        print("Không có ảnh nào được chọn. Thoát chương trình.")
        break

    # Xử lý ảnh được chọn
    process_image(img_path)

    # Hỏi người dùng có muốn chọn ảnh khác không
    repeat = input("Bạn có muốn chọn ảnh khác không? (y/n): ")
    if repeat.lower() != 'y':
        print("Thoát chương trình.")
        break
