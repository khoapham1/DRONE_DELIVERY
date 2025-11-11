# test_recording.py
import cv2
import os

import numpy as np

def test_video_writer():
    # Kiểm tra thư mục
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
        print("Created recordings directory")
    
    # Kiểm tra video writer
    filename = 'recordings/test_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
    
    if video_writer.isOpened():
        print(f"Video writer created successfully: {filename}")
        
        # Tạo test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Ghi vài frame
        for i in range(10):
            video_writer.write(test_frame)
        
        video_writer.release()
        print(f"Test video saved: {filename}")
        
        # Kiểm tra file
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"File size: {file_size} bytes")
        else:
            print("File not created!")
    else:
        print("Failed to create video writer")

if __name__ == "__main__":
    test_video_writer()