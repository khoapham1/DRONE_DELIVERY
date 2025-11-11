import cv2
import numpy as np
import glob

# Kích thước của bảng checkerboard (số góc bên trong)
CHECKERBOARD = (6, 9)  

# Tạo các mảng để lưu tọa độ điểm
objpoints = []  # Điểm 3D thực tế
imgpoints = []  # Điểm 2D trên ảnh

# Chuẩn bị tọa độ 3D của các góc trên bảng
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Đọc tất cả các ảnh chụp checkerboard
images = glob.glob('calibration_images/*.jpg')  # Đọc tất cả ảnh calib_1.jpg, calib_2.jpg, ...

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tìm các góc của bảng checkerboard
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Lưu ma trận camera và distortion để sử dụng sau này
np.save("camera_matrix_gpt.npy", mtx)
np.save("dist_coeff_gpt.npy", dist)

print("Calibration Successful!")
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)