from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from utils.Sliding_and_Pyramid import sliding_window
from utils.Sliding_and_Pyramid import image_pyramid
import numpy as np
import imutils
import time
import cv2
import argparse

# Khai báo cách nhập dữ liệu vào
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image.")
ap.add_argument("-s", "--size", default="(300, 200)",
                help="ROI size(px).")
ap.add_argument("-c", "--min-confi", type=float, default=0.9,
                help="minimum probility to filter weak dectections.")
ap.add_argument("-v", "--vsl", type=int, default=-1,
                help="Whether or not to show extra visualizations for debugging.")

args = vars(ap.parse_args())


# Khởi tạo các biến để sử dụng trong quá trình xử lý
WIDTH = 416
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

# Tải phân loại ResNet và ảnh đầu vào
print("[INFO] Loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# Tải ảnh từ disk, sau đó resize
orig = cv2.imread(args["image"])
orig = imutils.resize(image=orig, width=WIDTH)
(h, w) = orig.shape[:2]

# Khởi tạo image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, min_Size=ROI_SIZE)

# Khởi tại 2 lists, một cái để lưu ROIs generator từ kim tự tháp ảnh và cửa sổ trượt
# Cái còn lại để lưu trữ tọa độ (x, y) của các ROI trong ảnh gốc
rois = []
locs = []

# Thời gian bắt đầu vòng lặp
start = time.time()

# Lặp qua ảnh kim tự tháp
for image in pyramid:
    # Xem xét tỷ lệ giữa kích thước ảnh ban đầu và kích thước của các lớp trong khối kim tự tháp
    scale = w / float(image.shape[1])

    # Cho mỗi lớp trong khối kim tự tháp, và trượt cửa sổ trượt qua các vị trí
    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # Tỷ lệ (x, y) của ROI có liên quan tới kích thước của ảnh ban đầu
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        # Lấy ROI và xử lý chúng, sau đó cũng có lựa chọn vùng bằng các sử dụng Keras/Tensorflow
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        # Cập nhật danh sách của ROIs và các tâm
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        # Kiểm tra xem, nếu chúng ta hiển thị mỗi cửa sổ trượt trong khối kim tự tháp
        if args["vsl"] > 0:
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Hiển thị kết quả và ROI hiện tại
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

# Kiểm tra thời gian xử lý
end = time.time()
print("[INFO] Looping over pyramid/windows took {:.5f} seconds".format(end - start))

# Chuyển đổi ROIs sang 1 Numpy array
rois = np.array(rois, dtype="float32")

# Phân loại mỗi vùng ROIs sử dụng ResNet và sau đó kiểm tra xem mất bao nhiêu thời gian
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))

# Decode những dự đoán và khởi tạo một dictionary với class labels(keys) tới từng ROIs có label (value)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

# Dự đoán
for (i, p) in enumerate(preds):
    # Tạo mối liên quan giữa thông tin dự đoán cho ROI hiện tại
    (imagenetID, label, prob) = p[0]

    # Lọc những dự đoán kém để đảm bảo cho xác xuất dự đoán là tốt hơn khả năng dự đoán tối thiểu
    if prob < args["min_confi"]:
        # Boudding box với dự đoán và chuyển đổi sang các tọa độ
        box = locs[i]

        # Danh sách các dự đoán cho nhãn, thêm boudding box và xác xuất dự báo cho danh sách đó.
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L


# Lặp qua tất cả các keys và labels
for label in labels.keys():
    # Copy ảnh ban đầu để vẽ lên nó
    print("[INFO] showing results for '{}'".format(label))
    clone = orig.copy()

    # Lặp qua tất cả các boudding box cho nhãn hiện tại
    (startX, startY, endX, endY) = box
    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Show results
    cv2.imshow("Before", clone)
    clone = orig.copy()

    # Apply non-maximum Suppression

    # Giải nét các boudding box và xác xuất dự đoán liên quan, sau đó áp dụng NMS
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes=boxes, probs=proba)

    # Lặp qua tất cả các boudding boxes, cái mà đc giữ lại sau khi sử dụng NMS
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


    # SHOW
    cv2.imshow("After", clone)
    cv2.waitKey(0)














