#!/usr/bin/env python3

######## IMPORTS #########

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import cv2.aruco as aruco
import sys
import time
import math
import numpy as np
from cv_bridge import CvBridge
from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal
from pymavlink import mavutil
from array import array
from rclpy.executors import SingleThreadedExecutor

####### VARIABLES ########

vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1
vehicle.parameters['PLND_EST_TYPE'] = 0
vehicle.parameters['LAND_SPEED'] = 27  ## cms/s

velocity = -0.5  # m/s
takeoff_height = 6  # m
########################

# Initialize CvBridge for ROS2 image conversion
bridge = CvBridge()

ids_to_find = [1, 2]
marker_sizes = [50, 20]
marker_heights = [10, 3]

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

horizontal_res = 1280
vertical_res = 720

horizontal_fov = 62.2 * (math.pi / 180)  ## 62.2 for picam V2, 53.5 for V1
vertical_fov = 48.8 * (math.pi / 180)  ## 48.8 for V2, 41.41 for V1

found_count = 0
notfound_count = 0

############# CAMERA INTRINSICS #######

dist_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]
camera_matrix = [[530.8269276712998, 0.0, 320.5], [0.0, 530.8269276712998, 240.5], [0.0, 0.0, 1.0]]
np_camera_matrix = np.array(camera_matrix)
np_dist_coeff = np.array(dist_coeff)

#####
time_last = 0
time_to_wait = 0.1  ## 100 ms

time_to_sleep = 5  ## seconds the drone will wait after dropping off the taco
sub = None  ## initialize subscriber variable to make it global

def send_local_ned_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0,
        0,
        0,
        vx,
        vy,
        vz,
        0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_land_message(x,y):
    msg = vehicle.message_factory.landing_target_encode(
        0,
        0,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        x,
        y,
        0,0,0
        )
    vehicle.send_mavlink(msg)
    vehicle.flush()

class DroneNode(Node):
    def __init__(self):
        super().__init__('drone_node')
        self.newimg_pub = self.create_publisher(Image, '/UAV/forward/image_new', 10)
        self.subscription = self.create_subscription(
            Image,
            '/UAV/forward/image_raw',
            self.msg_receiver,
            10)
        # Add camera info subscriber
        self.camera_matrix = None
        self.dist_coeffs = None
    


    def msg_receiver(self, message):
        global notfound_count, found_count, time_last, time_to_wait, id_to_find, sub
        # simple low-pass last-angle storage (attach attributes to node)
        if not hasattr(self, 'last_x_ang'):
            self.last_x_ang = 0.0
            self.last_y_ang = 0.0
            self.lp_alpha = 0.5  # smoothing factor (0..1). Thấp hơn -> mượt hơn.

        if time.time() - time_last > time_to_wait:
            # convert image
            cv_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # detect markers (parameters created earlier)
            corners, ids, rejected = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)

            # altitude decide which marker to look for
            altitude = vehicle.location.global_relative_frame.alt  ## meters

            id_to_find = 0
            marker_height = 0
            marker_size = 0

            if altitude > marker_heights[1]:
                id_to_find = ids_to_find[0]
                marker_height = marker_heights[0]
                marker_size = marker_sizes[0]
            else:
                id_to_find = ids_to_find[1]
                marker_height = marker_heights[1]
                marker_size = marker_sizes[1]

            print("Looking for marker:", id_to_find)

            # set vehicle to LAND mode (non-blocking). DON'T block here.
            if vehicle.mode!='LAND':
                vehicle.mode = VehicleMode('LAND')
                while vehicle.mode!='LAND':
                    time.sleep(1)
                print("Vehicle is LAND mode.")

            # if vehicle disarmed, stop subscription
            if not vehicle.armed:
                try:
                    self.destroy_subscription(self.subscription)
                except Exception:
                    pass
                cv2.destroyAllWindows()
                return None

            found_id = False
            try:
                if ids is not None and len(ids) > 0:
                    # flatten ids to simple 1D array and iterate with index
                    ids_flat = ids.flatten()
                    for idx, marker_id in enumerate(ids_flat):
                        if int(marker_id) == int(id_to_find):
                            # extract corresponding corners for this marker
                            corners_single = [corners[idx]]
                            corners_single_np = np.asarray(corners_single)

                            # pose estimation (marker_size should be in meters for meaningful tvec)
                            # If your marker_sizes are in cm, convert to meters: marker_size_m = marker_size / 100.0
                            marker_size_m = marker_size / 100.0  # <-- adjust if your sizes are in cm
                            ret = aruco.estimatePoseSingleMarkers(corners_single, marker_size_m, cameraMatrix=np_camera_matrix, distCoeffs=np_dist_coeff)
                            (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])
                            x = float(tvec[0]); y = float(tvec[1]); z = float(tvec[2])

                            # compute centroid in pixels
                            x_sum = corners_single_np[0][0][0][0] + corners_single_np[0][0][1][0] + corners_single_np[0][0][2][0] + corners_single_np[0][0][3][0]
                            y_sum = corners_single_np[0][0][0][1] + corners_single_np[0][0][1][1] + corners_single_np[0][0][2][1] + corners_single_np[0][0][3][1]
                            x_avg = x_sum / 4.0
                            y_avg = y_sum / 4.0

                            # convert pixel offset to angle (radians)
                            x_ang = (x_avg - horizontal_res * 0.5) * horizontal_fov / horizontal_res
                            y_ang = (y_avg - vertical_res * 0.5) * vertical_fov / vertical_res

                            # send landing target angles repeatedly while marker visible
                            if vehicle.mode !='LAND':
                                vehicle.mode = VehicleMode('LAND')
                                while vehicle.mode !='LAND':
                                    time.sleep(1)
                                print('Vehicle in LAND mode')
                                send_land_message(x_ang,y_ang)
                            else:
                                send_land_message(x_ang,y_ang)                            

                            marker_position = f'MARKER POSITION: x={x:.2f} y={y:.2f} z={z:.2f}'
                            aruco.drawDetectedMarkers(cv_image, corners)
                            # draw axis using rvec,tvec (converted to float arrays)
                            cv2.drawFrameAxes(cv_image, np_camera_matrix, np_dist_coeff, rvec, tvec, 0.1)  # axis length in meters
                            cv2.putText(cv_image, marker_position, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)
                            print(marker_position)
                            found_count += 1
                            found_id = True
                            break
                    if not found_id:
                        notfound_count += 1

                else:
                    notfound_count += 1
            except Exception as e:
                print('Target likely not found or pose failed:', e)
                notfound_count += 1

            # show image (non-blocking)
            # cv2.imshow("Aruco Detection", cv_image)
            # cv2.waitKey(1)

            # publish modified image
            new_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.newimg_pub.publish(new_msg)
            time_last = time.time()
        else:
            return None
        



################FUNCTIONS###############
def get_distance_meters(targetLocation, currentLocation):
    dLat = targetLocation.lat - currentLocation.lat
    dLon = targetLocation.lon - currentLocation.lon

    return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5

def goto(targetLocation):
    distanceToTargetLocation = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)

    vehicle.simple_goto(targetLocation)

    while vehicle.mode.name == "GUIDED":
        currentDistance = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
        if currentDistance < distanceToTargetLocation * 0.02:
            print("Reached target waypoint")
            time.sleep(2)
            break
        time.sleep(1)
    return None

def arm_and_takeoff(targetHeight):
    while not vehicle.is_armable:
        print('Waiting for vehicle to become armable')
        time.sleep(1)
    print('Vehicle is now armable')

    vehicle.mode = VehicleMode('GUIDED')

    while vehicle.mode != 'GUIDED':
        print('Waiting for drone to enter GUIDED flight mode')
        time.sleep(1)
    print('Vehicle now in GUIDED mode. Have Fun!')

    vehicle.armed = True
    time.sleep(1)
    while not vehicle.armed:
        print('Waiting for vehicle to become armed.')
        time.sleep(1)
    print('Look out! Virtual props are spinning!')

    vehicle.simple_takeoff(targetHeight)

    while True:
        print('Current Altitude: %d' % vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= 0.95 * targetHeight:
            break
        time.sleep(1)
    print('Target altitude reached!') 

    return None

## Send velocity command to drone




def lander(node, duration=30):
    """
    Spin ROS2 node for 'duration' seconds to allow ArUco detection.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    start_time = time.time()
    try:
        while rclpy.ok() and time.time() - start_time < duration and vehicle.armed:
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.remove_node(node)

def main(args=None):
    rclpy.init(args=args)
    drone_node = DroneNode()
    
    try:
        ### Record home coordinates of drone so we know where to fly back to after delivery
        lat_home = vehicle.location.global_relative_frame.lat
        lon_home = vehicle.location.global_relative_frame.lon

        wp_home = LocationGlobalRelative(lat_home, lon_home, takeoff_height)
        wp_taco = LocationGlobalRelative(-35.36303741, 149.1652374, takeoff_height)  ## Original waypoint +25 meters

        arm_and_takeoff(takeoff_height)
        time.sleep(1)

        ############ Fly to taco dropoff waypoint
        goto(wp_taco)

        ############ Precision land on taco arucos
        print("Precision landing on taco arucos...")
        lander(drone_node, duration=60)
        print("")
        print("----------------------------------")
        print("Arrived at the taco destination")
        print("Dropping tacos and heading home.")
        print("-----------ENJOY------------------")
        time.sleep(time_to_sleep)

        ########### Fly the drone back to home waypoint
        arm_and_takeoff(takeoff_height)
        goto(wp_home)
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode != "LAND":
            print("Waiting for LAND mode...")
            vehicle.mode = VehicleMode("LAND")
            time.sleep(1)

        print("Precision landing on taco arucos...")
        lander(drone_node, duration=60)
        print("")
        print("----------------------------------")
        print("Made it home for another delivery")
        print("----------------------------------")

    except KeyboardInterrupt:
        pass
    finally:
        drone_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()