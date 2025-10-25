#!/usr/bin/env python3
# drone_control.py (updated without ROS2)
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
marker_sizes = [60, 10]          # cm
marker_heights = [10, 3]         # m, altitude thresholds
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.05
if hasattr(cv2.aruco, 'CORNER_REFINE_SUBPIX'):
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 30
    parameters.cornerRefinementMinAccuracy = 0.01

horizontal_res = 1280
vertical_res = 720
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

calib_path = "/home/pi/Delivery/drone_gps/"
np_camera_matrix = np.load(calib_path + 'camera_matrix_gpt.npy')
np_dist_coeff = np.load(calib_path + 'dist_coeff_gpt.npy')


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
        config = _picamera2.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        _picamera2.configure(config)
        _picamera2.start()
        
        _camera_running = True
        _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
        _camera_thread.start()
        print("Started Picamera2 successfully")
        
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
            
            # Convert BGR to RGB if needed and encode to JPEG
            if frame is not None:
                # Picamera2 returns RGB, convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ret, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
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
            _picamera2.close()
        except Exception as e:
            print("Error stopping camera:", e)
        _picamera2 = None
    
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
    def __init__(self, connection_str='udp:100.82.242.10:5000', takeoff_height=3):
        """
        Create DroneController and connect to vehicle.
        For Raspberry Pi with Pixhawk: '/dev/ttyS0,115200'
        For UDP (4G module): 'udp:100.82.242.10:14550'
        """
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)

        self.vehicle = connect(connection_str, baud=115200, wait_ready=True, timeout=120)
        
        # Set landing parameters
        try:
            self.vehicle.parameters['PLND_ENABLED'] = 1
            self.vehicle.parameters['PLND_TYPE'] = 1  # ArUco-based precision landing
            self.vehicle.parameters['PLND_EST_TYPE'] = 0
            self.vehicle.parameters['LAND_SPEED'] = 20
        except Exception:
            print("Failed to set some landing parameters")
            
        self.takeoff_height = takeoff_height
        self.flown_path = []  # Store actual flown path
        self.aruco_thread = None
        self.aruco_running = False
        # Danh sách ArUco IDs cần detect (mặc định)
        self.find_aruco = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    
    def start_image_stream(self, topic_name=None):
        """
        Start camera for image streaming
        """
        try:
            start_camera()
            print("Started camera stream")
        except Exception as e:
            print("Failed to start camera:", e)
            
    def stop_image_stream(self):
        """
        Stop camera stream
        """
        try:
            stop_camera()
            print("Stopped camera stream")
        except Exception as e:
            print("Failed to stop camera:", e)
    
    # Method để set danh sách find_aruco động
    def set_find_aruco(self, ids):
        if isinstance(ids, list) and all(isinstance(id, int) for id in ids):
            self.find_aruco = ids
        else:
            raise ValueError("Invalid ArUco IDs. Must be list of integers.")

    def send_aruco_marker_to_server(self, markers):
        """
        Send detected ArUco markers to server endpoint.
        Use localhost since server runs on same Raspberry Pi
        """
        try:
            response = requests.post('http://localhost:5000/update_aruco_markers', 
                                   json={'markers': markers}, timeout=2)
            if response.status_code == 200:
                print("Successfully sent ArUco markers to server")
            else:
                print(f"Failed to send ArUco markers, status code: {response.status_code}") 
        except Exception as e:
            print(f"Error sending ArUco markers to server: {e}")

    # MAVLink helpers
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def send_land_message(self, x, y):
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
        print(f"Set speed to {speed} m/s")

    # Core flight primitives
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=60, speed=2.5):
        """
        Goto with increased tolerance and timeout to avoid stuck, record position.
        """
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
        # while not self.vehicle.is_armable:
        #     print('Waiting for vehicle to become armable')
        #     time.sleep(1)
        
        # self.vehicle.mode = VehicleMode('GUIDED')
        while self.vehicle.mode != 'GUIDED':
            print('Waiting for GUIDED mode...')
            time.sleep(1)
        
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)
        
        print("Drone is armed and ready")
        return True

    def arm_and_takeoff(self, targetHeight):
        while not self.vehicle.is_armable:
            print('Waiting for vehicle to become armable')
            time.sleep(1)
        # self.vehicle.mode = VehicleMode('GUIDED')
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

            print('Altitude: %.2f' % (alt if alt else 0.0))
            if alt >= 0.95 * targetHeight:
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
            print("ArUco processing already running")
            return
            
        self.aruco_running = True
        self.aruco_thread = threading.Thread(target=self._aruco_processing_loop, daemon=True)
        self.aruco_thread.start()
        print("Started ArUco marker detection")

    def _aruco_processing_loop(self):
        """
        Main loop for ArUco marker detection - chạy liên tục
        """
        global time_last
        
        # Initialize CLAHE for image enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
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
                    
                detected_markers = self._process_frame_for_aruco(cv_image, clahe)
                
                # Send detected markers to server
                if detected_markers:
                    try:
                        self.send_aruco_marker_to_server(detected_markers)
                    except Exception as e:
                        print("Error sending ArUco marker:", e)
                        
            except Exception as e:
                print("ArUco processing error:", e)

                time.sleep(0.1)
    def _process_frame_for_aruco(self, cv_image, clahe):
        """
        Process a single frame for ArUco marker detection
        """
        detected_markers = {}
        
        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Preprocess to enhance marker contrast
        try:
            clahe_img = clahe.apply(gray_img)
        except Exception:
            clahe_img = gray_img
        blur = cv2.GaussianBlur(clahe_img, (3,3), 0)

        # Detect markers
        corners, ids, rejected = aruco.detectMarkers(blur, aruco_dict, parameters=parameters)

        # Process detected markers
        if ids is not None:
            ids_flat = ids.flatten()
            for idx, marker_id in enumerate(ids_flat):
                marker_id = int(marker_id)
                
                # Check if this is a marker we want to find
                if marker_id in self.find_aruco:
                    marker_size = 60  # cm
                    ret = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                        cameraMatrix=np_camera_matrix,
                                                        distCoeffs=np_dist_coeff)
                    rvec, tvec = ret[0][idx][0, :], ret[1][idx][0, :]

                    x = float(tvec[0])
                    y = float(tvec[1])
                    z = float(tvec[2])
                    
                    marker_position = f'MARKER DETECTED - ID: {marker_id}, POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                    print(marker_position)
                    try:
                        cv2.drawFraneAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)
                        aruco.drawDetectMarker(cv_image, corners)
                    except Exception as e:
                        pass
                    cv2.putText(cv_image, marker_position, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)                    
                    # Only record if marker is within reasonable range
                    if -50 < y < 50:  # z should be positive
                        try:
                            lat = self.vehicle.location.global_relative_frame.lat
                            lon = self.vehicle.location.global_relative_frame.lon
                            detected_markers[marker_id] = [lat, lon]
                            print(f"Recorded marker ID {marker_id} at lat={lat:.6f}, lon={lon:.6f}")
                        except Exception as e:
                            print(f"Error getting location for marker {marker_id}: {e}")

        return detected_markers

    def _process_landing_marker(self, marker_id, corners, idx, cv_image):
        """
        Process landing markers (IDs 1 and 2)
        """
        altitude = self.vehicle.location.global_relative_frame.alt
        if altitude is None:
            altitude = 0.0
            
        if altitude > marker_heights[1]:
            id_to_find = ids_to_find[0]
            marker_size = marker_sizes[0]
        else:
            id_to_find = ids_to_find[1]
            marker_size = marker_sizes[1]

        if marker_id == id_to_find:
            corners_single = [corners[idx]]
            marker_size_m = marker_size / 100.0
            
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size,
                                                  cameraMatrix=np_camera_matrix,
                                                  distCoeffs=np_dist_coeff)
            rvec, tvec = ret[0][idx][0, :], ret[1][idx][0, :]

            x = float(tvec[0])
            y = float(tvec[1])
            z = float(tvec[2])

            x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
            y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]

            x_avg = x_sum * 0.25
            y_avg = y_sum * 0.25

            x_ang = (x_avg - horizontal_res * 0.5) * horizontal_fov / horizontal_res
            y_ang = (y_avg - vertical_res * 0.5) * vertical_fov / vertical_res

            self.send_land_message(x_ang, y_ang)
            print(f"Sending landing target x_ang={x_ang:.3f}, y_ang={y_ang:.3f}")

            # Draw visualization
            marker_position = f'LANDING MARKER: x={x:.2f} y={y:.2f} z={z:.2f}'
            try:
                cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)
            except Exception:
                pass
            print(marker_position)

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

    def fly_and_precision_land_with_waypoints(self, waypoints, loiter_alt=3, aruco_duration=30):
        """
        Fly to waypoints while detecting ArUco markers, return via interpolated recorded path, precision land at home.
        """
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
        print(f"Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly to waypoints [1:-1] (skip start, exclude goal)
        for i, wp in enumerate(waypoints[1:-1]):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Fly to final goal
        goal_wp = waypoints[-1]
        wp_target = LocationGlobalRelative(goal_wp[0], goal_wp[1], loiter_alt)
        print("Flying to final target", goal_wp[0], goal_wp[1])
        self.goto(wp_target)

        # Stop ArUco detection
        self.stop_aruco_processing()

        # Land
        print("Starting landing phase...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != "LAND":
            print("Waiting for LAND mode...")
            time.sleep(1)

        # Wait until disarmed
        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)
            
        print("Mission complete")

# Utility to create controller
_controller = None

def get_controller(connection_str='udp:100.82.242.10:5000', takeoff_height=3):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller