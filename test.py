import torch
import cv2
import matplotlib.pyplot as plt

# Tải mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Đọc ảnh
img_path = 'cat.png'  # Đường dẫn đến ảnh của bạn
img = cv2.imread(img_path)

# Phát hiện đối tượng
results = model(img)

# Hiển thị kết quả phát hiện
results.show()  # Hiển thị ảnh với bounding box của các đối tượng

# Trích xuất thông tin đối tượng phát hiện được
detections = results.pandas().xyxy[0]  # Lấy kết quả dưới dạng DataFrame

# Lọc các đối tượng là động vật (nếu YOLOv5 nhận dạng được)
animals = detections[detections['name'].isin(['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])]

# In thông tin của các loài động vật được phát hiện
for idx, row in animals.iterrows():
    print(f"Đối tượng: {row['name']}, Xác suất: {row['confidence']:.2f}, Vị trí: [{row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}]")
