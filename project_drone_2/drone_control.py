#!/usr/bin/env python3
# drone_control.py (record and interpolate flown path for exact return trajectory
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

# ---- camera / aruco params (copy from yours) ---
aruco = cv2.aruco
bridge = CvBridge()
ids_to_find = [1, 2]
marker_sizes = [60, 10]          # cm
marker_heights = [10, 3]         # m, altitude thresholds
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

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

horizontal_res = 1280
vertical_res = 720
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

# ---- DroneController class ----

def send_land_message(vehicle, x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0, 0, mavutil.mavlink.MAV_FRAME_BODY_FRD,
        x, y, 0, 0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

class DroneController:

    def __init__(self, connection_str='tcp:127.0.0.1:5763', takeoff_height=6):
        """
        Create DroneController and connect to vehicle.
        """
        self.connection_str = connection_str
        print("Connecting to vehicle on", connection_str)
        self.vehicle = connect(connection_str, wait_ready=True, timeout=120)
        # set some landing params
        try:
            self.vehicle.parameters['PLND_ENABLED'] = 1
            self.vehicle.parameters['PLND_TYPE'] = 1  # ArUco-based precision landing
            self.vehicle.parameters['PLND_EST_TYPE'] = 0
            self.vehicle.parameters['LAND_SPEED'] = 20

        except Exception:
            print("Failed to set some landing parameters")
        self.takeoff_height = takeoff_height
        self.ros_node = None
        self.flown_path = []  # Store actual flown path

    # MAVLink helpers
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    # Core flight primitives
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

    def goto(self, targetLocation, tolerance=2.0, timeout=30):
        """
        Goto with increased tolerance and timeout to avoid stuck, record position.
        """
        distanceToTargetLocation = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
        self.vehicle.simple_goto(targetLocation)
        start_dist = distanceToTargetLocation
        start_time = time.time()
        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(targetLocation, self.vehicle.location.global_relative_frame)
            # Record current position
            current_pos = self.vehicle.location.global_relative_frame
            if current_pos.lat and current_pos.lon:
                self.flown_path.append([current_pos.lat, current_pos.lon])
            print(f"Distance to target: {currentDistance:.2f}m (threshold: {max(tolerance, start_dist * 0.01):.2f}m)")
            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.5)
        print("Timeout reaching waypoint, proceeding anyway")
        return False

    def arm_and_takeoff(self, targetHeight):
        while not self.vehicle.is_armable:
            print('Waiting for vehicle to become armable')
            time.sleep(1)
        self.vehicle.mode = VehicleMode('GUIDED')
        while self.vehicle.mode != 'GUIDED':
            print('Waiting for GUIDED...')
            time.sleep(1)
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)
        self.vehicle.simple_takeoff(targetHeight)
        start_time = time.time()
        while time.time() - start_time < 12:
            alt = self.vehicle.location.global_relative_frame.alt
            print('Altitude: %.2f' % (alt if alt else 0.0))
            if alt is not None and alt >= 0.9 * targetHeight:
                print("Reached takeoff altitude")
                break
            time.sleep(1)
        else:
            print("Takeoff timeout, proceeding at current altitude")

    def _start_aruco_node_and_lander(self, duration=30):
        """
        Start ROS2 node that subscribes to camera image and publishes landing commands while spinning for 'duration'
        """
        if not rclpy.ok():
            rclpy.init(args=None)
        node = DroneNode(self.vehicle)  # Pass the vehicle to DroneNode
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        start_time = time.time()
        try:
            # Stay in GUIDED mode during precision landing
            self.vehicle.mode = VehicleMode("GUIDED")
            while self.vehicle.mode != "GUIDED":
                print("Waiting for GUIDED mode for precision landing...")
                time.sleep(1)
            while rclpy.ok() and time.time() - start_time < duration and self.vehicle.armed:
                executor.spin_once(timeout_sec=0.1)
                # Check altitude to switch to LAND mode
                alt = self.vehicle.location.global_relative_frame.alt
                if alt is not None and alt < 1.0:  # PLND_ALT_LOW
                    print("Low altitude reached, switching to LAND mode")
                    self.vehicle.mode = VehicleMode("LAND")
                    while self.vehicle.mode != "LAND":
                        print("Waiting for LAND mode...")
                        time.sleep(1)
                    break
        finally:
            executor.remove_node(node)

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

    def fly_and_precision_land_with_waypoints(self, waypoints, loiter_alt=5, aruco_duration=30):
        """
        Fly to delivery point, record path, land, takeoff, return via interpolated recorded path.
        """
        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        # Clear previous flown path
        self.flown_path = []

        # Takeoff from home
        print("Arming and taking off")
        self.arm_and_takeoff(loiter_alt)
        time.sleep(1)

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

        # Precision land with ArUco
        print("Starting precision landing phase (aruco)...")
        self._start_aruco_node_and_lander(duration=aruco_duration)
        print("Delivery landing complete")

        # Ensure disarmed and in GUIDED mode after landing
        time.sleep(5)
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode != "GUIDED":
            print("Waiting for GUIDED after landing...")
            time.sleep(1)
        if self.vehicle.armed:
            self.vehicle.armed = False
            while self.vehicle.armed:
                print("Disarming after landing...")
                time.sleep(1)

        # Interpolate recorded path for return
        return_path = self.interpolate_path(self.flown_path[::-1], num_points=20)  # Reverse and interpolate
        print("Interpolated return path:", return_path)

        # Takeoff for return
        print("Re-arming and taking off for return home")
        self.arm_and_takeoff(loiter_alt+2)
        time.sleep(1)

        # Fly back via interpolated return path
        print("Returning to home via recorded path")
        for i, wp in enumerate(return_path):
            wp_loc = LocationGlobalRelative(wp[0], wp[1], loiter_alt)
            print(f"Flying back to waypoint {i+1}: {wp[0]}, {wp[1]}")
            self.goto(wp_loc)

        # Final LAND at home
        print("Starting precision landing phase at home (aruco)...")
        self._start_aruco_node_and_lander(duration=30)
        print("Mission complete, landed at home")


class DroneNode(Node):
    def __init__(self, vehicle):
        node_name = f"drone_node_for_aruco{int(time.time()*1000)}"
        super().__init__(node_name)
        self.vehicle = vehicle  # Store the vehicle object
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        self.subscription = self.create_subscription(Image, '/UAV/forward/image_raw', self.msg_receiver, 10)

        # tracking state (we keep simple smoothing but no lost-handling logic)
        self.last_detection_time = 0.0
        self.last_x_ang = 0.0
        self.last_y_ang = 0.0
        self.lp_alpha = 0.6  # low-pass smoothing for x_ang,y_ang

        # image preprocessing helpers
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def preprocess(self, gray_img):
        """
        Preprocess image to improve ArUco detectability:
         - CLAHE (local histogram equalization)
         - small gaussian blur
        """
        try:
            clahe_img = self.clahe.apply(gray_img)
        except Exception:
            clahe_img = gray_img

        blur = cv2.GaussianBlur(clahe_img, (3,3), 0)
        return blur

    def msg_receiver(self, message):
        global time_last
        # throttle processing
        if time.time() - time_last < time_to_wait:
            return
        time_last = time.time()

        try:
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Preprocess to enhance marker contrast
            preproc = self.preprocess(gray_img)

            # detect markers on preprocessed image first
            corners, ids, rejected = aruco.detectMarkers(preproc, aruco_dict, parameters=parameters)
            # if not detected, fallback to original gray (sometimes helps)
            if ids is None or len(ids) == 0:
                corners2, ids2, rejected2 = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
                if ids2 is not None and len(ids2) > 0:
                    corners, ids, rejected = corners2, ids2, rejected2

            altitude = self.vehicle.location.global_relative_frame.alt
            if altitude is None:
                altitude = 0.0

            # choose marker id by altitude
            if altitude > marker_heights[1]:
                id_to_find = ids_to_find[0]
                marker_size = marker_sizes[0]
            else:
                id_to_find = ids_to_find[1]
                marker_size = marker_sizes[1]

            found = False
            if ids is not None:
                ids_flat = ids.flatten()
                for idx, marker_id in enumerate(ids_flat):
                    if int(marker_id) == int(id_to_find):
                        # we found the marker we want
                        corners_single = [corners[idx]]
                        corners_single_np = np.asarray(corners_single)

                        # estimate pose (size in meters)
                        marker_size_m = marker_size / 100.0
                        ret = aruco.estimatePoseSingleMarkers(corners_single, marker_size_m,
                                                              cameraMatrix=np_camera_matrix,
                                                              distCoeffs=np_dist_coeff)
                        (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])
                        x = float(tvec[0]); y = float(tvec[1]); z = float(tvec[2])

                        # compute angles in body frame using pixel positions
                        cx = (corners_single_np[0][0][:,0].mean())
                        cy = (corners_single_np[0][0][:,1].mean())
                        x_ang = ((cx) - horizontal_res * 0.5) * horizontal_fov / horizontal_res
                        y_ang = ((cy) - vertical_res * 0.5) * vertical_fov / vertical_res

                        # Update last known angles
                        self.last_x_ang = x_ang
                        self.last_y_ang = y_ang

                        # send landing target
                        send_land_message(self.vehicle, x_ang, y_ang)

                        # annotate for visualization
                        marker_position = f'MARKER POS: x={x:.2f} y={y:.2f} z={z:.2f}'
                        aruco.drawDetectedMarkers(cv_image, corners)
                        try:
                            cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)
                        except Exception:
                            pass
                        cv2.putText(cv_image, marker_position, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)
                        print(marker_position)

                        self.last_detection_time = time.time()
                        found = True
                        break

            # If marker not detected, send last known landing target if recent
            if not found and (time.time() - self.last_detection_time) < 2.0:
                send_land_message(self.vehicle, self.last_x_ang, self.last_y_ang)
                cv2.putText(cv_image, "Using last known marker position", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            elif not found:
                cv2.putText(cv_image, "Marker not detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            # publish modified image for visualization if needed
            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)

        except Exception as e:
            # swallow for safety but print
            print("ArUco processing error:", e)

# Utility to create controller
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=6):
    global _controller
    if _controller is None:
        _controller = DroneController(connection_str=connection_str, takeoff_height=takeoff_height)
    return _controller