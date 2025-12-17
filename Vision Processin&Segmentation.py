import cv2
import numpy as np

# ===============================
# 1. Đọc ảnh công tơ
# ===============================
img = cv2.imread("data/Test1.jpg")
if img is None:
    raise ValueError("Không đọc được ảnh")

orig = img.copy()

# ===============================
# 2. Grayscale + tăng tương phản
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# ===============================
# 3. Edge Detection (Canny)
# ===============================
edges = cv2.Canny(gray, 100, 200)

# ===============================
# 4. Tìm ROI hiển thị số
# ===============================
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

roi_candidates = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    ratio = w / float(h)

    # Khung số thường dài ngang
    if 0.8 < ratio < 1.5 and area > 20000:
        roi_candidates.append((x, y, w, h))

if len(roi_candidates) == 0:
    raise ValueError("Không tìm thấy ROI")

# Lấy ROI lớn nhất
x, y, w, h = max(roi_candidates, key=lambda b: b[2] * b[3])
roi = gray[y:y+h, x:x+w]

# ===============================
# 5. Deskew (chỉnh thẳng)
# ===============================
coords = np.column_stack(np.where(roi > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h_roi, w_roi) = roi.shape
M = cv2.getRotationMatrix2D((w_roi // 2, h_roi // 2), angle, 1.0)
roi = cv2.warpAffine(
    roi, M, (w_roi, h_roi),
    flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE
)

# ===============================
# 6. Nhị phân hóa + làm sạch
# ===============================
_, thresh = cv2.threshold(
    roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# ===============================
# 7. Tách từng chữ số
# ===============================
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

digits = []

for cnt in contours:
    x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)

    # Lọc nhiễu
    if h_d > 0.5 * thresh.shape[0] and w_d > 10:
        digit = thresh[y_d:y_d+h_d, x_d:x_d+w_d]
        digit = cv2.resize(digit, (28, 28))
        digits.append((x_d, digit))

# Sắp xếp từ trái sang phải
digits = sorted(digits, key=lambda x: x[0])
digit_images = [d[1] for d in digits]

# ===============================
# 8. Hiển thị kết quả
# ===============================
cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Original + ROI", orig)
cv2.imshow("ROI", roi)
cv2.imshow("Threshold", thresh)

for i, d in enumerate(digit_images):
    cv2.imshow(f"Digit {i}", d)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Đã tách được {len(digit_images)} chữ số")
