#!/usr/bin/env python3
import time
import math
import threading
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from picamera2 import Picamera2
import requests

# ---- camera / aruco params ----
aruco = cv2.aruco
ids_to_find = [1, 2]

# find_aruco is now dynamic, will be set via class
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # Use legacy API to avoid segfault
parameters = cv2.aruco.DetectorParameters_create()  # Use legacy API to avoid segfault



horizontal_res = 1280
vertical_res = 720
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

calib_path = "/home/pi/Project_drone_delivery-main/"
np_camera_matrix = np.load(calib_path + 'camera_matrix_gpt_pi2.npy')
np_dist_coeff = np.load(calib_path + 'dist_coeff_gpt_pi2.npy')

time_to_wait = 0.1
time_last = 0

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_picamera2 = None
_camera_thread = None
_camera_running = False

# ---- Camera Functions ----
def start_camera():
    """
    Start Picamera2 and continuously capture frames
    """
    global _picamera2, _camera_running, _camera_thread
    
    if _camera_running:
        return
    
    try:
        _picamera2 = Picamera2()
        
        # Cấu hình camera với các thông số màu sắc chính xác
        config = _picamera2.create_video_configuration(
            main={
                "size": (1280, 720), 
                "format": "RGB888"
            },
            controls={
                # "AwbEnable": True,           # Bật cân bằng trắng tự động
                # "AwbMode": 0,               # Auto white balance mode
                # "Brightness": 0,          # Độ sáng trung tính
                # "Contrast": 1,            # Tăng độ tương phản nhẹ
                # "Saturation": 0.5,          # Độ bão hòa trung tính
                # "Sharpness": 0,           # Độ sắc nét trung tính
                # "ExposureTime": 20000,      # Tăng thời gian phơi sáng để cải thiện ánh sáng
                # "AnalogueGain": 0    # Tăng gain nhẹ để cải thiện độ sáng
                # "ExposureValue": -0.5
            }
        )
        _picamera2.configure(config)
        _picamera2.start()
        
        _camera_running = True
        _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
        _camera_thread.start()
        print("Started Picamera2 successfully with optimized color settings")
        
    except Exception as e:
        print("Failed to start Picamera2:", e)

def _camera_loop():
    """
    Continuously capture frames and update latest frame
    """
    global _latest_frame_jpeg, _latest_frame_lock, _camera_running
    
    while _camera_running and _picamera2:
        try:
            # Capture frame
            frame = _picamera2.capture_array()
            
            if frame is not None:               
                # Convert JPEG with High Quality
                ret, jpeg = cv2.imencode('.jpg', frame, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 90,  # Tăng chất lượng JPEG
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ])
                if ret:
                    with _latest_frame_lock:
                        _latest_frame_jpeg = jpeg.tobytes()

            time.sleep(0.033)  # ~30fps
        except Exception as e:
            print("Error in camera loop:", e)
            time.sleep(0.1)

def stop_camera():
    """
    Stop camera and clean up
    """
    global _picamera2, _camera_running, _camera_thread
    
    _camera_running = False
    if _camera_thread:
        _camera_thread.join(timeout=2.0)
    
    if _picamera2:
        try:
            _picamera2.stop()
            _picamera2 = None
        except Exception as e:
            print("Error stopping camera:", e)
    
    print("Stopped camera")

def get_lastest_frame():
    """
    Return latest JPEG bytes or None.
    """
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg

# ---- DroneController class ----

class DroneController:
    def __init__(self, connection_str='udp:192.168.137.152:5000', takeoff_height=3):
        """
        Create DroneController and connect to vehicle.
        """
        self.connection_str = 'udp:192.168.137.152:5000'
        print(" Connecting to vehicle on", connection_str)

        try:
            self.vehicle = connect(connection_str, baud=115200, wait_ready=True, timeout=120)
            print(" Vehicle connected successfully")
        except Exception as e:
            print(f" Failed to connect to vehicle: {e}")
            self.vehicle = None
        
        # Set landing parameters
        if self.vehicle:
            try:
                self.vehicle.parameters['PLND_ENABLED'] = 1
                self.vehicle.parameters['PLND_TYPE'] = 1  # ArUco-based precision landing
                self.vehicle.parameters['PLND_EST_TYPE'] = 0
                self.vehicle.parameters['LAND_SPEED'] = 20
                print("Landing parameters set successfully")
            except Exception as e:
                print(" Failed to set some landing parameters:", e)
            
        self.takeoff_height = takeoff_height
        self.flown_path = []  # Store actual flown path
        self.aruco_thread = None
        self.aruco_running = False
        # Danh sách ArUco IDs cần detect (mặc định)
        self.find_aruco = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        #Zone parameters
        self.zone_center_offset = [0, 0]
        # self.zone_radius = 300
        self.zone_width = 1280
        self.zone_height = 720
    def start_image_stream(self, topic_name=None):
        """
        Start camera for image streaming
        """
        try:
            start_camera()
            print(" Started camera stream")
        except Exception as e:
            print(" Failed to start camera:", e)
            
    def stop_image_stream(self):
        """
        Stop camera stream
        """
        try:
            stop_camera()
            print(" Stopped camera stream")
        except Exception as e:
            print(" Failed to stop camera:", e)
    
    # Method để set danh sách find_aruco động
    def set_find_aruco(self, ids):
        if isinstance(ids, list) and all(isinstance(id, int) for id in ids):
            self.find_aruco = ids
            print(f" Updated ArUco IDs to detect: {ids}")
        else:
            raise ValueError(" Invalid ArUco IDs. Must be list of integers.")

    def send_aruco_marker_to_server(self, markers):
        """
        Send detected ArUco markers to server endpoint.
        """
        try:
            response = requests.post('http://localhost:5000/update_aruco_markers', 
                                   json={'markers': markers}, timeout=2)
            if response.status_code == 200:
                print(" Successfully sent ArUco markers to server")
            else:
                print(f"Failed to send ArUco markers, status code: {response.status_code}") 
        except Exception as e:
            print(f" Error sending ArUco markers to server: {e}")

    # MAVLink helpers
    def send_local_ned_velocity(self, vx, vy, vz):
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def send_land_message(self, x, y):
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.landing_target_encode(
            0,
            0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            x,
            y,
            0,
            0,
            0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_speed(self, speed):
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0, 
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1, 0, 0, 0, 0 
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"✅ Set speed to {speed} m/s")

    # Core flight primitives
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=1.0):
        """
        Goto with increased tolerance and timeout to avoid stuck, record position.
        """
        if not self.vehicle:
            return False
            
        distanceToTargetLocation = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
        start_dist = distanceToTargetLocation
        start_time = time.time()
        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
            # Record current position
            current_pos = self.vehicle.location.global_relative_frame
            if current_pos.lat and current_pos.lon:
                self.flown_path.append([current_pos.lat, current_pos.lon])
            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.02)
        print("Timeout reaching waypoint, proceeding anyway")
        return False
    
    def arm_drone(self):
        """
        Arm the drone without taking off
        """
        if not self.vehicle:
            return False
            
        while self.vehicle.mode != 'GUIDED':
            print(' Waiting for GUIDED mode...')
            time.sleep(1)
        
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('⏳ Arming...')
            time.sleep(1)
        
        print("Drone is armed and ready")
        return True

    def arm_and_takeoff(self, targetHeight):
        if not self.vehicle:
            return
            
        while not self.vehicle.is_armable:
            print(' Waiting for vehicle to become armable')
            time.sleep(1)
            
        while self.vehicle.mode != 'GUIDED':
            print('Waiting for GUIDED...')
            time.sleep(1)
            
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)
            
        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt

            print(f'📊 Altitude: {alt:.2f}' if alt else 'Altitude: 0.00')
            if alt and alt >= 0.95 * targetHeight:
                break
            time.sleep(1)
        print("Reached takeoff altitude")
        return None
    
    def is_connected(self):
        """
        Check if vehicle is connected
        """
        try:
            return self.vehicle is not None and self.vehicle.location is not None
        except:
            return False

    def start_aruco_processing(self):
        """
        Start ArUco marker detection in separate thread
        """
        if self.aruco_running:
            print(" ArUco processing already running")
            return
            
        self.aruco_running = True
        self.aruco_thread = threading.Thread(target=self._aruco_processing_loop, daemon=True)
        self.aruco_thread.start()
        print("Started ArUco marker detection")

    def _aruco_processing_loop(self):

        global time_last
        
        print("ArUco processing started - ready to detect markers")
        
        while self.aruco_running:
            try:
                # Throttle processing
                if time.time() - time_last < time_to_wait:
                    time.sleep(0.01)
                    continue
                    
                time_last = time.time()
                
                # Get latest frame
                frame_jpeg = get_lastest_frame()
                if frame_jpeg is None:
                    time.sleep(0.01)
                    continue
                    
                # Convert JPEG to OpenCV image
                nparr = np.frombuffer(frame_jpeg, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if cv_image is None:
                    continue
                    
                detected_markers = self._process_frame_for_aruco(cv_image)
                
                # Send detected markers to server
                if detected_markers:
                    try:
                        self.send_aruco_marker_to_server(detected_markers)
                    except Exception as e:
                        print("Error sending ArUco marker:", e)
                        
            except Exception as e:
                print("ArUco processing error:", e)
                time.sleep(0.1)
    
    def _process_frame_for_aruco(self, cv_image):
        """
        Process a single frame for ArUco marker detection, only if within rectangular zone
        """
        detected_markers = {}
        
        # Chuyển sang grayscale
        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Áp dụng cân bằng histogram để cải thiện độ tương phản
        gray_img = cv2.equalizeHist(gray_img)
        
        # Làm mờ để giảm nhiễu
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0)

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(blur, aruco_dict, parameters=parameters)

        # Calculate camera center with offset
        center_x = cv_image.shape[1] // 2 + self.zone_center_offset[0]
        center_y = cv_image.shape[0] // 2 + self.zone_center_offset[1]

        # Calculate rectangle bounds
        left = center_x - self.zone_width // 2
        right = center_x + self.zone_width // 2
        top = center_y - self.zone_height // 2
        bottom = center_y + self.zone_height // 2

        # Vẽ rectangle zone cho visualization (tùy chọn, dù không stream ở đây)
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle
        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)  # Center dot

        # Process detected markers
        if ids is not None:
            ids_flat = ids.flatten()
            for idx, marker_id in enumerate(ids_flat):
                marker_id = int(marker_id)
                
                # Check if this is a marker we want to find
                if marker_id in self.find_aruco:
                    # Get marker center
                    corner = corners[idx][0]
                    marker_center_x = int(np.mean(corner[:, 0]))
                    marker_center_y = int(np.mean(corner[:, 1]))
                    
                    # Check if within rectangle zone
                    if left <= marker_center_x <= right and top <= marker_center_y <= bottom:
                        marker_size = 40  # cm
                        ret = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                            cameraMatrix=np_camera_matrix,
                                                            distCoeffs=np_dist_coeff)
                        rvec, tvec = ret[0][idx][0, :], ret[1][idx][0, :]
                        x = float(tvec[0])
                        y = float(tvec[1])
                        z = float(tvec[2])
                        
                        marker_position = f'MARKER DETECTED - ID: {marker_id}, POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                        print(f"🎯 {marker_position}")
                        
                        try:
                            cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)
                            aruco.drawDetectedMarkers(cv_image, corners)
                        except Exception as e:
                            print(f" Error drawing marker: {e}")
                        
                        # Vẽ line từ marker center đến camera center
                        cv2.line(cv_image, (marker_center_x, marker_center_y), (center_x, center_y), (0, 0, 255), 2)
                        
                        # Vẽ thông tin marker lên ảnh
                        cv2.putText(cv_image, marker_position, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)
                        
                        # Only record if marker is within reasonable range
                        if -50 < y < 50:  
                            # Kiểm tra xem có lấy được vị trí drone không
                            if self.vehicle and self.vehicle.location.global_relative_frame:
                                try:
                                    lat = self.vehicle.location.global_relative_frame.lat
                                    lon = self.vehicle.location.global_relative_frame.lon
                                    # Đảm bảo tọa độ hợp lệ
                                    if lat != 0 and lon != 0:
                                        detected_markers[marker_id] = [lat, lon]
                                        print(f" Recorded marker ID {marker_id} at lat={lat:.6f}, lon={lon:.6f}")
                                    else:
                                        print(f" Invalid GPS coordinates for marker {marker_id}")
                                except Exception as e:
                                    print(f"Error getting location for marker {marker_id}: {e}")

        return detected_markers

    def stop_aruco_processing(self):
        """
        Stop ArUco processing thread
        """
        self.aruco_running = False
        if self.aruco_thread:
            self.aruco_thread.join(timeout=2.0)
        print("Stopped ArUco processing")

    def interpolate_path(self, path, num_points=20):
        """
        Interpolate the recorded path to generate a smooth set of waypoints.
        """
        if not path or len(path) < 2:
            return path
        path = np.array(path) 
        t = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_points)
        lat = np.interp(t_new, t, path[:, 0])
        lon = np.interp(t_new, t, path[:, 1])
        return [[lat[i], lon[i]] for i in range(num_points)]

    def fly_and_precision_land_with_waypoints(self, waypoints, loiter_alt=4, aruco_duration=30):
        """
        Fly to waypoints while detecting ArUco markers
        """
        if not self.vehicle:
            print(" No vehicle connected")
            return
            
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff from home
        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

        # Start ArUco detection
        self.start_aruco_processing()

        # Store home
        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, loiter_alt)
        print(f" Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly to waypoints [1:-1] (skip start, exclude goal)
        for i, wp in enumerate(waypoints[1:-1]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f" Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Fly to final goal
        goal_wp = waypoints[-1]
        wp_target = LocationGlobalRelative(goal_wp[0], goal_wp[1], loiter_alt)
        print(f"Flying to final target {goal_wp[0]}, {goal_wp[1]}")
        self.goto(wp_target)

        # Stop ArUco detection
        self.stop_aruco_processing()

        # Land
        print(" Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != "LAND":
            print(" Waiting for LAND mode...")
            time.sleep(1)

        # Wait until disarmed
        while self.vehicle.armed:
            print(" Waiting for disarming...")
            time.sleep(1)
            
        print(" Mission complete")

_controller = None

def get_controller(connection_str='udp:192.168.137.152:5000', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller

# Thêm hàm cleanup cho Picamera2
def cleanup_camera():
    """Cleanup camera resources"""
    global _picamera2, _camera_running
    _camera_running = False
    if _picamera2:
        try:
            _picamera2.stop()
            # Removed _picamera2.close() to avoid double-close with __del__
            _picamera2 = None
        except Exception as e:
            print("Error stopping camera:", e)

# Đăng ký cleanup
import atexit
atexit.register(cleanup_camera)