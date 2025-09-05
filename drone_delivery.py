#!/usr/bin/env python3
import cv2
import time
import math
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

######## CONNECT VEHICLE ########
print("Connecting to vehicle...")
vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)

vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1
vehicle.parameters['PLND_EST_TYPE'] = 0
vehicle.parameters['LAND_SPEED'] = 35  # cm/s

######## VARIABLES ########
takeoff_height = 6  # m
time_to_sleep = 5   # seconds at drop point

ids_to_find = [1, 2]
marker_sizes = [60, 20]     # cm
marker_heights = [10, 3]    # m

# ---- ArUco compatibility (OpenCV 4.5/4.6 vs 4.7+) ----
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

if hasattr(aruco, 'ArucoDetector'):  # OpenCV 4.7+
    detector_params = aruco.DetectorParameters()
    _detector = aruco.ArucoDetector(aruco_dict, detector_params)
    def detect_markers(gray):
        return _detector.detectMarkers(gray)
else:  # OpenCV <= 4.6
    # Some builds only have DetectorParameters_create(); some accept DetectorParameters()
    detector_params = (aruco.DetectorParameters_create()
                       if hasattr(aruco, 'DetectorParameters_create')
                       else aruco.DetectorParameters())
    def detect_markers(gray):
        return aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

horizontal_res, vertical_res = 640, 480
horizontal_fov = 62.2 * (math.pi / 180)
vertical_fov = 48.8 * (math.pi / 180)

camera_matrix = np.array([[530.8269, 0.0, 320.5],
                          [0.0, 530.8269, 240.5],
                          [0.0, 0.0, 1.0]])
dist_coeff = np.zeros((5, 1))

######## DRONE FUNCTIONS ########
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

def arm_and_takeoff(targetHeight):
    while not vehicle.is_armable:
        print('Waiting for vehicle to become armable...')
        time.sleep(1)

    vehicle.mode = VehicleMode('GUIDED')
    while vehicle.mode.name != 'GUIDED':
        print('Waiting for drone to enter GUIDED mode...')
        time.sleep(1)

    vehicle.armed = True
    while not vehicle.armed:
        print('Waiting for vehicle to arm...')
        time.sleep(1)

    vehicle.simple_takeoff(targetHeight)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"Altitude: {alt:.2f} m")
        if alt >= 0.95 * targetHeight:
            print("Target altitude reached.")
            break
        time.sleep(1)

def send_land_message(x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0, 0, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        float(x), float(y), 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

######## VISION-BASED LANDING ########
def precision_land(camera_id=0, timeout=30):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, horizontal_res)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vertical_res)

    start_time = time.time()
    while time.time() - start_time < timeout and vehicle.armed:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray)

        altitude = vehicle.location.global_relative_frame.alt
        if altitude > marker_heights[1]:
            id_to_find = ids_to_find[0]
            marker_size = marker_sizes[0]
        else:
            id_to_find = ids_to_find[1]
            marker_size = marker_sizes[1]

        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            for i, marker_id in enumerate(ids_flat):
                if marker_id == id_to_find:
                    ret = aruco.estimatePoseSingleMarkers(
                        [corners[i]], marker_size, camera_matrix, dist_coeff)
                    (rvec, tvec) = (ret[0][0, 0, :], ret[1][0, 0, :])

                    # Pixel center of marker
                    x_avg = np.mean([pt[0] for pt in corners[i][0]])
                    y_avg = np.mean([pt[1] for pt in corners[i][0]])

                    # Convert pixel offset to angle (rad)
                    x_ang = (x_avg - horizontal_res * 0.5) * horizontal_fov / horizontal_res
                    y_ang = (y_avg - vertical_res * 0.5) * vertical_fov / vertical_res

                    send_land_message(x_ang, y_ang)
                    aruco.drawDetectedMarkers(frame, corners)
                    aruco.drawAxis(frame, camera_matrix, dist_coeff, rvec, tvec, 10)
                    print(f"Landing target: x={x_ang:.2f}, y={y_ang:.2f}")
                    break  # track 1 marker per loop

        cv2.imshow("Precision Landing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

######## MAIN ########
def main():
    print("Starting drone delivery mission...")

    # Save home position
    lat_home = vehicle.location.global_relative_frame.lat
    lon_home = vehicle.location.global_relative_frame.lon
    wp_home = LocationGlobalRelative(lat_home, lon_home, takeoff_height)

    # Taco delivery waypoint
    wp_taco = LocationGlobalRelative(-35.36303741, 149.1652374, takeoff_height)

    # 1. Takeoff
    arm_and_takeoff(takeoff_height)
    time.sleep(1)

    # 2. Fly to taco waypoint
    goto(wp_taco)

    # 3. Precision landing at taco
    vehicle.mode = VehicleMode("LAND")
    precision_land(camera_id=0, timeout=30)
    print("---- Taco delivered ----")
    time.sleep(time_to_sleep)

    # 4. Takeoff again
    arm_and_takeoff(takeoff_height)
    time.sleep(1)

    # 5. Fly back home
    goto(wp_home)

    # 6. Precision land at home
    vehicle.mode = VehicleMode("LAND")
    precision_land(camera_id=0, timeout=30)
    print("---- Drone back home ----")

if __name__ == "__main__":
    main()
