#!/usr/bin/env python3
"""
drone_delivery_node.py
ROS2 Humble Python node for precision landing using ArUco markers and DroneKit.
Camera topic: /camera/color/image_raw
Executable: ros2 run drone_delivery drone_delivery_node
"""

import threading
import time
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

class PrecisionLandingNode(Node):
    def __init__(self, vehicle,
                 camera_topic: str = '/camera/color/image_raw',
                 publish_topic: str = '/camera/color/image_new',
                 horizontal_res: int = 640,
                 vertical_res: int = 480,
                 horizontal_fov_rad: float = 62.2 * (math.pi / 180),
                 vertical_fov_rad: float = 48.8 * (math.pi / 180),
                 ids_to_find=(1,2),
                 marker_sizes_cm=(60,20),
                 marker_heights_m=(10,3),
                 process_period_s: float = 0.1):
        super().__init__('precision_landing_node')

        self.vehicle = vehicle
        self.camera_topic = camera_topic
        self.publish_topic = publish_topic

        self.horizontal_res = horizontal_res
        self.vertical_res = vertical_res
        self.horizontal_fov = horizontal_fov_rad
        self.vertical_fov = vertical_fov_rad

        self.ids_to_find = ids_to_find
        self.marker_sizes = marker_sizes_cm
        self.marker_heights = marker_heights_m

        self.process_period = process_period_s
        self._last_time = 0.0

        # Detection active flag (only process images when True)
        self.detection_active = False

        # Last detection info
        self.last_seen_time = 0.0
        self.last_seen_id = None
        self.last_tvec = None

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.publish_topic, 10)

        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.KEEP_LAST
        self.subscription = self.create_subscription(Image, self.camera_topic, self.image_callback, qos)

        aruco = cv2.aruco
        self.aruco = aruco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        # compatibility for OpenCV versions
        if hasattr(aruco, 'ArucoDetector'):
            self.detector_params = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            self._use_new_api = True
            self.get_logger().info("Using OpenCV ArUcoDetector (new API).")
        else:
            if hasattr(aruco, 'DetectorParameters_create'):
                self.detector_params = aruco.DetectorParameters_create()
            else:
                self.detector_params = aruco.DetectorParameters()
            self.detector = None
            self._use_new_api = False
            self.get_logger().info("Using older ArUco API (detectMarkers).")

        # camera intrinsics (replace with your calib if available)
        self.camera_matrix = np.array([[530.8269, 0.0, horizontal_res/2.0],
                                       [0.0, 530.8269, vertical_res/2.0],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeff = np.zeros((5, 1))

        self.get_logger().info(f"PrecisionLandingNode ready. Subscribed to {self.camera_topic}")

    def activate_detection(self, on: bool = True):
        """Enable/disable image processing in the subscriber callback."""
        self.detection_active = bool(on)
        if on:
            self.get_logger().info("ArUco detection ACTIVATED.")
            # reset last seen
            self.last_seen_time = 0.0
            self.last_seen_id = None
            self.last_tvec = None
        else:
            self.get_logger().info("ArUco detection DEACTIVATED.")

    def detect_markers(self, gray_img):
        if self._use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray_img)
            return corners, ids, rejected
        else:
            corners, ids, rejected = self.aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.detector_params)
            return corners, ids, rejected

    def send_land_message(self, x_ang_rad, y_ang_rad):
        try:
            msg = self.vehicle.message_factory.landing_target_encode(
                0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                float(x_ang_rad), float(y_ang_rad), 0.0, 0.0, 0.0)
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
        except Exception as e:
            self.get_logger().error(f"Failed to send landing_target MAVLink message: {e}")

    def image_callback(self, img_msg):
        # Only process images when detection is active
        if not self.detection_active:
            return

        now = time.time()
        if now - self._last_time < self.process_period:
            return
        self._last_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect_markers(gray)

        altitude = None
        try:
            altitude = float(self.vehicle.location.global_relative_frame.alt)
        except Exception:
            altitude = None

        if altitude is not None:
            if altitude > self.marker_heights[1]:
                id_to_find = self.ids_to_find[0]
                marker_size = self.marker_sizes[0]
            else:
                id_to_find = self.ids_to_find[1]
                marker_size = self.marker_sizes[1]
        else:
            id_to_find = self.ids_to_find[0]
            marker_size = self.marker_sizes[0]

        found = False
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                if int(marker_id) == int(id_to_find):
                    # Found target marker
                    try:
                        rvec_tvec = self.aruco.estimatePoseSingleMarkers([corners[i]], marker_size, self.camera_matrix, self.dist_coeff)
                        rvec = rvec_tvec[0][0][0,:]
                        tvec = rvec_tvec[1][0][0,:]
                    except Exception as e:
                        self.get_logger().warning(f"estimatePose failed: {e}")
                        rvec = None
                        tvec = None

                    pts = corners[i][0]
                    x_avg = float(np.mean(pts[:,0]))
                    y_avg = float(np.mean(pts[:,1]))

                    x_ang = (x_avg - self.horizontal_res * 0.5) * self.horizontal_fov / self.horizontal_res
                    y_ang = (y_avg - self.vertical_res * 0.5) * self.vertical_fov / self.vertical_res

                    # Store last seen info for main to use
                    self.last_seen_time = time.time()
                    self.last_seen_id = int(marker_id)
                    self.last_tvec = tvec if tvec is not None else None

                    try:
                        if self.vehicle.mode.name == 'LAND' or self.vehicle.mode.name == 'RTL':
                            self.send_land_message(x_ang, y_ang)
                            self.get_logger().info(f"Sent landing_target id={marker_id} x_ang={x_ang:.4f} y_ang={y_ang:.4f}")
                    except Exception as e:
                        self.get_logger().warning(f"Error while sending landing msg: {e}")

                    try:
                        self.aruco.drawDetectedMarkers(cv_image, corners)
                        if rvec is not None and tvec is not None:
                            self.aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeff, rvec, tvec, 10)
                    except Exception:
                        pass

                    found = True
                    break

        # publish annotated image (even if not found)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.pub.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge publish error: {e}")

        if not found:
            # keep last_seen_time unchanged (so main can detect timeout)
            pass


# ---------------- mission helpers ----------------
def arm_and_takeoff(vehicle, target_height):
    while not vehicle.is_armable:
        time.sleep(1.0)
    vehicle.mode = VehicleMode('GUIDED')
    while vehicle.mode.name != 'GUIDED':
        time.sleep(0.5)
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(0.5)
    vehicle.simple_takeoff(target_height)
    while True:
        alt = float(vehicle.location.global_relative_frame.alt)
        if alt >= target_height * 0.95:
            break
        time.sleep(0.5)

def goto_waypoint(vehicle, targetLocation, timeout=300):
    def distance(a,b):
        return math.sqrt((a.lat - b.lat)**2 + (a.lon - b.lon)**2) * 1.113195e5
    initial_distance = distance(targetLocation, vehicle.location.global_relative_frame)
    vehicle.simple_goto(targetLocation)
    start = time.time()
    while vehicle.mode.name == 'GUIDED':
        curdist = distance(targetLocation, vehicle.location.global_relative_frame)
        if curdist < initial_distance * 0.02:
            break
        if time.time() - start > timeout:
            break
        time.sleep(1.0)

# ---------------- main ----------------
def main():
    rclpy.init()

    # connect to vehicle (modify connection string if needed)
    node = None
    print("Connecting to vehicle tcp:127.0.0.1:5763 ...")
    vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True, heartbeat_timeout=60)

    # set land params if available
    try:
        vehicle.parameters['PLND_ENABLED'] = 1
        vehicle.parameters['PLND_TYPE'] = 1
        vehicle.parameters['PLND_EST_TYPE'] = 0
        vehicle.parameters['LAND_SPEED'] = 35
    except Exception:
        pass

    # save home and taco waypoint
    lat_home = float(vehicle.location.global_relative_frame.lat)
    lon_home = float(vehicle.location.global_relative_frame.lon)
    wp_home = LocationGlobalRelative(lat_home, lon_home, 6)
    wp_taco = LocationGlobalRelative(-35.36303741, 149.1652374, 6)

    # create node and spin thread
    node = PrecisionLandingNode(vehicle)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    node.get_logger().info("PrecisionLandingNode spinning in background.")

    try:
        node.get_logger().info("Arm & takeoff...")
        arm_and_takeoff(vehicle, 6)
        time.sleep(1.0)

        node.get_logger().info("Going to taco waypoint...")
        goto_waypoint(vehicle, wp_taco)

        node.get_logger().info("Switch to LAND for precision landing...")

        # Activate image detection so node will process camera frames
        node.activate_detection(True)

        # switch to LAND mode to enable precision landing messages
        vehicle.mode = VehicleMode('LAND')

        # Wait for the node to see the marker within timeout (e.g. 30s)
        detect_timeout = 30.0
        detect_start = time.time()
        node.get_logger().info(f"Waiting up to {detect_timeout}s for ArUco marker...")
        seen = False
        while time.time() - detect_start < detect_timeout and vehicle.armed:
            if node.last_seen_time > 0 and (time.time() - node.last_seen_time) < 2.0:
                node.get_logger().info(f"Marker {node.last_seen_id} detected recently (tvec={node.last_tvec}). Continuing to wait for disarm.")
                seen = True
                break
            time.sleep(0.2)

        if not seen:
            node.get_logger().warning("Marker not detected within timeout. Continuing to wait for landing (disarm) without visual assist.")

        # Now wait until drone is disarmed (land complete)
        node.get_logger().info("Waiting until vehicle disarmed (landing completes)...")
        while vehicle.armed:
            time.sleep(0.5)

        node.get_logger().info("Vehicle disarmed -> landing finished.")

        # Deactivate detection
        node.activate_detection(False)

        node.get_logger().info("Taco delivered. Sleeping a bit...")
        time.sleep(5.0)

        node.get_logger().info("Takeoff for return...")
        arm_and_takeoff(vehicle, 6)

        node.get_logger().info("Going home...")
        goto_waypoint(vehicle, wp_home)

        node.get_logger().info("Final LAND for home precision landing...")

        # Activate detection for home landing
        node.activate_detection(True)
        vehicle.mode = VehicleMode('LAND')

        # same detection wait
        detect_timeout = 30.0
        detect_start = time.time()
        seen = False
        while time.time() - detect_start < detect_timeout and vehicle.armed:
            if node.last_seen_time > 0 and (time.time() - node.last_seen_time) < 2.0:
                node.get_logger().info(f"Marker {node.last_seen_id} detected recently at home (tvec={node.last_tvec}).")
                seen = True
                break
            time.sleep(0.2)

        if not seen:
            node.get_logger().warning("Home marker not detected within timeout. Continuing to wait for landing (disarm).")

        node.get_logger().info("Waiting until vehicle disarmed (landing completes)...")
        while vehicle.armed:
            time.sleep(0.5)

        node.get_logger().info("Mission complete.")

    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down")
    finally:
        try:
            node.activate_detection(False)
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        try:
            vehicle.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
