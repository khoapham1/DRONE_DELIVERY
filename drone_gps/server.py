# server.py (updated without ROS2)
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading, time
from drone_control import get_controller, get_lastest_frame
import json
from planner import run_planner  # Import planner
from dronekit import VehicleMode
import math
from dronekit import LocationGlobalRelative
import cv2
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create/connect controller singleton
# For UDP connection over 4G:
controller = get_controller(connection_str='udp:100.82.242.10:5000', takeoff_height=3)

# controller = get_controller(connection_str='udp:100.101.89.55:14550', takeoff_height=3)

prev_loc = None
distance_traveled = 0.0

# Lưu trữ thông tin ArUco markers
aruco_markers = {}

# Start image streamer so we always have frame to serve
try:
    controller.start_image_stream()
    controller.start_aruco_processing()
except Exception as e:
    print("Warning: Failed to start image streamer:", e)

def mjpeg_generator():
    """
    Generator that MJPEG frames with ArUco marker detection overlay.
    """
    while True:
        frame = get_lastest_frame()
        if frame is None:
            # Placeholder nếu không có frame (hình đen 1x1)
            placeholder = cv2.imencode('.jpg', np.zeros((1,1,3), np.uint8))[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)  # Giảm rate nếu no frame
            continue
        
        try:
            # Convert JPEG bytes to OpenCV image
            nparr = np.frombuffer(frame, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Process ArUco marker detection on the frame
                processed_frame = process_aruco_on_frame(cv_image)
                
                # Encode the processed frame back to JPEG
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    frame = jpeg.tobytes()
        
        except Exception as e:
            print(f"Error processing frame for ArUco: {e}")
            # If processing fails, use original frame
            pass
        
        # multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30fps

def process_aruco_on_frame(cv_image):
    """
    Process ArUco marker detection on a frame and draw detection results.
    """
    try:
        # Get current ArUco markers from global storage
        global aruco_markers
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Initialize ArUco dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()
        
        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # If markers are detected, draw them
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            # Draw IDs and information
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id[0])
                corner = corners[i][0]
                
                # Calculate center position for text
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                
                # Draw marker ID
                cv2.putText(cv_image, f"ID: {marker_id}", 
                           (center_x - 30, center_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Check if we have GPS coordinates for this marker
                if str(marker_id) in aruco_markers:
                    gps_coords = aruco_markers[str(marker_id)]
                    cv2.putText(cv_image, 
                               f"Lat: {gps_coords[0]:.6f}", 
                               (center_x - 30, center_y + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.putText(cv_image, 
                               f"Lon: {gps_coords[1]:.6f}", 
                               (center_x - 30, center_y + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add status text to frame
        cv2.putText(cv_image, f"ArUco Markers Detected: {len(ids) if ids is not None else 0}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(cv_image, f"Total Known: {len(aruco_markers)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"Error in ArUco frame processing: {e}")
    
    return cv_image
# --- VIDEO ---
@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/start_aruco_detection', methods=['POST'])
def start_aruco_detection():
    try:
        controller.start_aruco_processing()
        return jsonify({'status': 'success', 'message': 'ArUco detection started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_aruco_detection', methods=['POST'])
def stop_aruco_detection():
    try:
        controller.stop_aruco_processing()
        return jsonify({'status': 'success', 'message': 'ArUco detection stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# _______________________________________________________________________
def telemetry_loop():
    global prev_loc, distance_traveled
    while True:
        try:
            # Kiểm tra kết nối vehicle thay vì controller.is_connected()
            if not controller.vehicle or not controller.vehicle.location:
                socketio.emit('telemetry', {'error': 'Not connected to vehicle'})
                time.sleep(2)
                continue
                
            v = controller.vehicle
            # Sử dụng global_relative_frame thay vì global_frame để tránh None
            loc = v.location.global_relative_frame
            if loc.lat is None or loc.lon is None:
                time.sleep(0.5)
                continue
                
            current_loc = (loc.lat, loc.lon)
            velocity = v.groundspeed if v.groundspeed else 0.0
            
            # Tính toán distance traveled
            if prev_loc:
                dlat = current_loc[0] - prev_loc[0]
                dlon = current_loc[1] - prev_loc[1]
                dist_step = math.sqrt((dlat**2) + (dlon**2)) * 1.113195e5  # meters
                distance_traveled += dist_step
            prev_loc = current_loc
            
            data = {
                'lat': loc.lat,
                'lon': loc.lon,
                'alt': loc.alt,
                'mode': str(v.mode.name),
                'velocity': velocity,
                'distance_traveled': distance_traveled
            }
            socketio.emit('telemetry', data)
        except Exception as e:
            print("Error in telemetry loop:", e)
            socketio.emit('telemetry', {'error': str(e)})
        time.sleep(0.5)

# Endpoint để cập nhật ArUco markers từ drone
@app.route('/update_aruco_markers', methods=['POST'])
def update_aruco_markers():
    try:
        payload = request.get_json(silent=True) or {}
        markers = payload.get('markers', {})
        
        print(f"Received ArUco markers: {markers}")
        
        # Cập nhật markers toàn cục
        global aruco_markers
        aruco_markers.update(markers)
        
        # Gửi realtime đến tất cả clients
        socketio.emit('aruco_markers_update', {'markers': aruco_markers})
        
        return jsonify({'status': 'success', 'markers_received': len(markers)})
    except Exception as e:
        print(f"Error updating ArUco markers: {e}")
        return jsonify({'error': str(e)}), 500



# Endpoint để lấy danh sách ArUco markers hiện tại
@app.route('/get_aruco_markers', methods=['GET'])
def get_aruco_markers():
    return jsonify(aruco_markers)

# Endpoint để set danh sách ArUco IDs cần detect
@app.route('/set_aruco_ids', methods=['POST'])
def set_aruco_ids():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get('ids', [])
        if not isinstance(ids, list) or not all(isinstance(id, int) for id in ids):
            return jsonify({'error': 'Invalid IDs format. Must be list of integers.'}), 400
        
        # Cập nhật find_aruco trong controller
        controller.set_find_aruco(ids)
        
        print(f"Updated find_aruco to: {ids}")
        return jsonify({'status': 'success', 'ids': ids})
    except Exception as e:
        print(f"Error setting ArUco IDs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_gps_stations', methods=['GET'])
def get_gps_stations():
    try:
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start full mission (can pass {"station": "station1"} or {"station": "station2"})
@app.route('/start_mission', methods=['POST'])
def start_mission():
    global distance_traveled
    distance_traveled = 0.0  # Reset distance
    
    try:
        payload = request.get_json(silent=True) or {}
        station_name = payload.get('station', 'station1')
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)

        if station_name not in data:
            return jsonify({'error': f'station "{station_name}" not found in JSON'}), 400

        # Lấy home position làm start
        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        
        # Tạo waypoints từ home đến tất cả các điểm trong JSON (theo station được chọn)
        waypoints = [start]
        for point in data[station_name]:
            waypoints.append([point['lat'], point['lon']])
        
        # Gửi planned path đến web
        socketio.emit('planned_path', {'waypoints': waypoints})
        
        # Start mission in background với waypoints
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'station': station_name, 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Fly selected points
@app.route('/fly_selected', methods=['POST'])
def fly_selected():
    global distance_traveled
    distance_traveled = 0.0  # Reset distance
    
    try:
        payload = request.get_json(silent=True) or {}
        points = payload.get('points', [])
        
        if not points:
            return jsonify({'error': 'No points selected'}), 400
            
        # Lấy home position làm start
        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        
        # Tạo waypoints từ home đến các điểm được chọn
        waypoints = [start]
        for point in points:
            waypoints.append([point['lat'], point['lon']])
        
        # Gửi planned path đến web
        socketio.emit('planned_path', {'waypoints': waypoints})
        
        # Start mission in background với waypoints
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/return_home', methods=['POST'])
def return_home():
    try:
        v = controller.vehicle
        home_lat = v.location.global_frame.lat
        home_lon = v.location.global_frame.lon
        home_alt = 3  # meters
        home_location = LocationGlobalRelative(home_lat, home_lon, home_alt)
        v.simple_goto(home_location)
        return jsonify({'status': 'returning_home', 'home': [home_lat, home_lon]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_mission_in_thread(waypoints):
    def mission():
        try:
            socketio.emit('mission_status', {'status': 'starting', 'waypoints': waypoints})
            controller.fly_and_precision_land_with_waypoints(waypoints, loiter_alt=2.3, aruco_duration=60)
            socketio.emit('mission_status', {'status': 'completed'})
        except Exception as e:
            socketio.emit('mission_status', {'status': 'error', 'error': str(e)})
    t = threading.Thread(target=mission, daemon=True)
    t.start()
    return t

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear_aruco_markers', methods=['POST'])
def clear_aruco_markers():
    global aruco_markers
    aruco_markers = {}
    socketio.emit('aruco_markers_update', {'markers': aruco_markers})
    return jsonify({'status': 'cleared'})

@app.route('/fly', methods=['POST'])
def fly_route():
    global distance_traveled
    distance_traveled = 0.0
    payload = request.json
    lat = payload.get('lat')
    lon = payload.get('lon')
    if lat is None or lon is None:
        return jsonify({'error': 'missing lat/lon'}), 400
    v = controller.vehicle
    start = [v.location.global_frame.lat, v.location.global_frame.lon]
    goal = [lat, lon]
    planner_payload = {"start": start, "goal": goal}
    waypoints = run_planner(planner_payload)
    socketio.emit('planned_path', {'waypoints': waypoints})
    run_mission_in_thread(waypoints)
    return jsonify({'status': 'mission_started', 'waypoints': waypoints})

@socketio.on('change_mode')
def handle_change_mode(data):
    mode = data.get('mode')
    if mode in ['LAND', 'GUIDED']:
        try:
            controller.vehicle.mode = VehicleMode(mode)
            emit('mission_status', {'status': f'mode_changed_to_{mode}'})
        except Exception as e:
            emit('mission_status', {'status': 'error', 'error': str(e)})
    else:
        emit('mission_status', {'status': 'invalid_mode'})

@app.route('/arm', methods=['POST'])
def arm_drone():
    try:
        success = controller.arm_drone()
        if success:
            return jsonify({'status': 'armed', 'message': 'Drone armed successfully'})
        else:
            return jsonify({'error': 'Failed to arm drone'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000)
