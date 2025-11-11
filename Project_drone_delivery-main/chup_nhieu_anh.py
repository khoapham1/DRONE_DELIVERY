from picamera2 import Picamera2
import cv2
import os
import time
import datetime

# Tạo thư mục lưu ảnh calibration
if not os.path.exists("calibration_images"):
    os.makedirs("calibration_images")

# Khởi tạo camera
picam2 = Picamera2()



# Cấu hình preview (xem trước)
preview_config = picam2.create_preview_configuration(
    main={"size": (1024, 768)}
)
picam2.configure(preview_config)

# Bắt đầu camera
picam2.start()

# Biến đếm số ảnh đã chụp
image_count = 0

try:
    while image_count < 40:
        # Đọc frame từ camera
        frame = picam2.capture_array()

        # Chuyển đổi màu từ RGB sang BGR (OpenCV sử dụng BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Hiển thị frame
        cv2.imshow("Camera Live - Adjust Chessboard", frame)

        # Chờ phím nhấn
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím 'c' để chụp ảnh
        if key == ord('c'):
            # Tạo tên tệp duy nhất với thời gian
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.abspath(f"calibration_images/image_{timestamp}_{image_count + 1}.jpg")
            
            # Lưu ảnh
            success = cv2.imwrite(image_name, frame)
            if success:
                print(f"Đã lưu ảnh: {image_name}")
                print(f"Kích thước ảnh: {frame.shape}")  # Kiểm tra kích thước ảnh
                image_count += 1
            else:
                print(f"Lỗi khi lưu ảnh: {image_name}")

            # Hiển thị thông báo trên frame
            cv2.putText(frame, f"Captured: {image_count}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Camera Live - Adjust Chessboard", frame)
            cv2.waitKey(500)  # Hiển thị thông báo trong 0.5 giây

        # Nhấn phím 'q' để thoát
        elif key == ord('q'):
            break
finally:
    # Dừng camera và đóng cửa sổ
    picam2.stop()
    cv2.destroyAllWindows()
    print("Hoàn thành chụp ảnh calibration.")