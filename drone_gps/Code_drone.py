
#!/usr/bin/env python3
import time
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
from rclpy.executors import SingleThreadedExecutor

# ---- camera / aruco params ---
takeoff_height = 6
lat_pkg = 10.85091904
lon_pkg = 106.77125894
bridge = CvBridge()
ids_to_find = [1, 2]
marker_sizes = [60, 10]          # cm
marker_heights = [10, 3]         # m, altitude thresholds
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# globals aruco node/executor/thread
aruco_node = None
aruco_executor = None
aruco_thread = None
aruco_start_lock = threading.Lock()

# lock cho vehicle
vehicle_lock = threading.Lock()
# Tuning detector parameters for robustness
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

horizontal_res = 640
vertical_res = 480
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

dist_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]
camera_matrix = [[530.8269276712998, 0.0, 320.5],
                 [0.0, 530.8269276712998, 240.5],
                 [0.0, 0.0, 1.0]]

np_camera_matrix = np.array(camera_matrix)
np_dist_coeff = np.array(dist_coeff)

time_to_wait = 0.1
time_last = 0
# initialize globals used in node
first_run = 0
start_time = 0
notfound_count = 0
found_count = 0

# ----- Drone Node ----
class DroneNode(Node):
    def __init__(self, vehicle):
        node_name = f"drone_node_for_aruco{int(time.time()*1000)}"
        super().__init__(node_name)
        self.vehicle = vehicle
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.lander, 10)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, gray_img):
        """ Preprocess the image to improve marker detection. """
        try:
            clahe_img = self.clahe.apply(gray_img)
        except Exception:
            clahe_img = gray_img

        blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        return blur

    def lander(self, message):
        global time_last, first_run, start_time, notfound_count, found_count
        if first_run == 0:
            print("Initializing Landing system")
            start_time = time.time()
            first_run = 1
        time_last = time.time()

        cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        preproc = self.preprocess(gray_img)

        # detect markers on preprocessed image
        corners, ids, rejected = detector.detectMarkers(image=preproc)

        # Ensure vehicle in LAND mode before sending landing target
        if self.vehicle.mode is None or self.vehicle.mode.name != 'LAND':
            print("Switching vehicle to LAND mode...")
            self.vehicle.mode = VehicleMode("LAND")
            timeout = time.time() + 10
            while (self.vehicle.mode is None or self.vehicle.mode.name != 'LAND') and time.time() < timeout:
                print(' Waiting for drone to enter LAND mode...')
                time.sleep(0.5)

        counter = 0
        altitude = self.vehicle.location.global_relative_frame.alt
        if altitude is None:
            print("Warning: Altitude data unavailable, using default marker")
            id_to_find = ids_to_find[1]
            marker_size = marker_sizes[1]
        elif altitude > marker_heights[1]:
            id_to_find = ids_to_find[0]
            marker_height = marker_heights[0]
            marker_size = marker_sizes[0]
        else:
            id_to_find = ids_to_find[1]
            marker_size = marker_sizes[1]

        print(f"Looking for marker {id_to_find}, marker sizes list: {marker_sizes}")

        try:
            if ids is not None:
                for idx in range(len(ids)):
                    id_val = int(ids[idx][0]) if hasattr(ids[idx], '__iter__') else int(ids[idx])
                    if id_val == id_to_find:
                        corners_single = [corners[idx]]
                        ret = cv2.aruco.estimatePoseSingleMarkers(corners_single, marker_size, np_camera_matrix, np_dist_coeff)
                        (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])
                        x = float(tvec[0])
                        y = float(tvec[1])
                        z = float(tvec[2])

                        # Center pixel calculation for this marker
                        corner = corners[idx][0]
                        x_sum = corner[0][0] + corner[1][0] + corner[2][0] + corner[3][0]
                        y_sum = corner[0][1] + corner[1][1] + corner[2][1] + corner[3][1]

                        x_avg = x_sum * 0.25
                        y_avg = y_sum * 0.25

                        x_ang = (x_avg - horizontal_res*0.5) * (horizontal_fov/horizontal_res)
                        y_ang = (y_avg - vertical_res*0.5) * (vertical_fov/vertical_res)  

                        send_land_message(x_ang, y_ang)

                        marker_position = f'MARKER POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                        cv2.aruco.drawDetectedMarkers(cv_image, [corners[idx]])
                        cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 10)
                        print("X Center pixel: "+str(x_avg)+" Y Center pixel: "+str(y_avg))
                        print("Found count: "+str(found_count)+ " Notfound count: "+str(notfound_count))
                        print(marker_position)
                        found_count += 1
                    counter += 1
            else:
                notfound_count += 1

            # publish modified image for visualization 
            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.newimg_pub.publish(new_msg)

        except Exception as e:
            print("Aruco processing error: ", e)
            notfound_count += 1

# ---- Drone helper functions ----
def send_local_ned_velocity(vehicle, vx, vy, vz):
    """ Move vehicle in direction based on specified velocity vectors. """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        vx, vy, vz, # m/s
        0, 0, 0, # accelerations (not supported)
        0, 0)    # yaw, yaw_rate (not supported)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_land_message(x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0,  # time_boot_ms
        0,  # target_num
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  #
        x,  # x angle (radians)
        y,  # y angle (radians)
        0,  # distance (not used)
        0,  # size_x (not used)
        0   # size_y (not used)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# GPS helpers
def get_distance_meters(targetLocation, currentLocation):
    dlat = targetLocation.lat - currentLocation.lat
    dlon = targetLocation.lon - currentLocation.lon
    return math.sqrt((dlon * dlon) + (dlat * dlat)) * 1.113195e5

def goto(targetLocation, tolerance=2.0, timeout=30):
    distanceToTargetLocation = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
    vehicle.simple_goto(targetLocation)
    start_dist = distanceToTargetLocation
    start_time = time.time()
    while vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
        currentDistance = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
        current_pos = vehicle.location.global_relative_frame
        if current_pos.lat and current_pos.lon:
            flow_path.append((current_pos.lat, current_pos.lon))
        print(f"Distance to target: {currentDistance:.2f} m (threshold: {max(tolerance, start_dist*0.02):.2f} m)")
        if currentDistance < max(tolerance, start_dist*0.01):
            print("Reached target waypoint.")
            return True
        time.sleep(0.5)
    print("time out reaching waypoint, proceeding anyway")
    return False

def arm_and_takeoff(targetHeight):
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    timeout = time.time() + 10
    while (vehicle.mode is None or vehicle.mode.name != 'GUIDED') and time.time() < timeout:
        print(" Waiting for vehicle to enter GUIDED mode...")
        time.sleep(0.5)

    vehicle.armed = True
    arm_timeout = time.time() + 15
    while not vehicle.armed and time.time() < arm_timeout:
        print(" Waiting for arming...")
        time.sleep(0.5)

    if not vehicle.armed:
        raise RuntimeError("Failed to arm vehicle within timeout")

    print(f"Taking off to {targetHeight} meters")
    vehicle.simple_takeoff(targetHeight)
    takeoff_timeout = time.time() + 30
    while True:
        altitude = vehicle.location.global_relative_frame.alt or 0.0
        print(f" Altitude: {altitude:.1f} m")
        if altitude >= targetHeight * 0.95 or time.time() > takeoff_timeout:
            break
        time.sleep(1)

    if vehicle.location.global_relative_frame.alt < targetHeight * 0.95:
        print("Warning: did not reach target altitude within timeout.")
    else:
        print("Reached target altitude.")
    return None

if __name__ == '__main__':
    global vehicle
    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True, timeout=120)
    
    rclpy.init()
    ros_node = DroneNode(vehicle)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_node)

    try:
        vehicle.parameters['PLND_ENABLED'] = 1
        vehicle.parameters['PLND_TYPE'] = 1
        vehicle.parameters['PLND_EST_TYPE'] = 0
        vehicle.parameters['LAND_SPEED'] = 25
        print("Precision landing parameters set successfully")
        print(f"PLND_ENABLED: {vehicle.parameters['PLND_ENABLED']}")
        print(f"PLND_TYPE: {vehicle.parameters['PLND_TYPE']}")
        print(f"PLND_EST_TYPE: {vehicle.parameters['PLND_EST_TYPE']}")
        print(f"LAND_SPEED: {vehicle.parameters['LAND_SPEED']}")
    except Exception as e:
        print("Error setting params:", e)

    lat_home = vehicle.location.global_relative_frame.lat
    lon_home = vehicle.location.global_relative_frame.lon

    wp_home = LocationGlobalRelative(lat_home, lon_home, takeoff_height)
    wp_pkg = LocationGlobalRelative(lat_pkg, lon_pkg, takeoff_height)
    flow_path = []

    arm_and_takeoff(takeoff_height)
    goto(wp_pkg)
    
    while vehicle.armed and rclpy.ok():
        executor.spin_once(timeout_sec=0.033)
        time.sleep(0.033)

    print("")
    print("----------------------------------")
    print("Arrived at the taco destination!")
    print("Dropping tacos and heading home.")
    print("----------ENJOY!----------------")

    arm_and_takeoff(takeoff_height)
    goto(wp_home)
    while vehicle.armed and rclpy.ok():
        executor.spin_once(timeout_sec=0.033)
        time.sleep(0.033)

    print("")
    print("----------------------------------")
    print("Made it home for another delivery!")
    print("----------------------------------")

    ros_node.destroy_node()
    rclpy.shutdown()
    vehicle.close()
