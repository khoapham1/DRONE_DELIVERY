# server.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import time
from drone_control import get_controller, get_lastest_frame
import json
from planner import run_planner  # Import planner
from dronekit import VehicleMode, LocationGlobalRelative
import math
import cv2
import numpy as np
import signal
import sys
import atexit
import socket
import os
import datetime
from threading import Lock

app = Flask(__name__)
# Sửa CORS settings để cho phép kết nối từ mọi nguồn
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   logger=False, 
                   engineio_logger=False,
                   async_mode='threading')

# Create/connect controller singleton
try:
    controller = get_controller(connection_str='udp:192.168.137.152:5000', takeoff_height=3)
    print(" Drone controller initialized successfully")
except Exception as e:
    print(f"Failed to initialize drone controller: {e}")
    controller = None

prev_loc = None
distance_traveled = 0.0
# Record start time
recording = False
recording_lock = Lock()
video_writer = None
recording_start_time = None

# Thêm variables global (hoặc trong hàm nếu muốn)
zone_center_offset = [0, 0]  # Offset
zone_width = 1280
zone_height = 720

# Lưu trữ thông tin ArUco markers
aruco_markers = {}

# Start image streamer so we always have frame to serve
try:
    if controller:
        controller.start_image_stream()
        controller.start_aruco_processing()
        print(" Camera and ArUco processing started successfully")
    else:
        from drone_control import start_camera
        start_camera()
        print("Camera started successfully")
except Exception as e:
    print(" Warning: Failed to start image streamer:", e)

def get_network_info():
    """Lấy thông tin mạng chi tiết"""
    try:
        hostname = socket.gethostname()
        all_ips = []
        for interface in socket.getaddrinfo(hostname, None):
            ip = interface[4][0]
            if ip not in all_ips and not ip.startswith('127.'):
                all_ips.append(ip)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            main_ip = s.getsockname()[0]
            s.close()
        except:
            main_ip = all_ips[0] if all_ips else "0.0.0.0"
        return hostname, main_ip, all_ips
    except Exception as e:
        return "unknown", "0.0.0.0", []

def check_port_availability(port=5000):
    """Kiểm tra xem port có sẵn sàng không"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('0.0.0.0', port))
        sock.close()
        return result == 0
    except:
        return False

def cleanup():
    """Cleanup function to be called on exit"""
    print(" Cleaning up resources...")
    try:
        if controller:
            controller.stop_aruco_processing()
            controller.stop_image_stream()
    except Exception as e:
        print("Error during cleanup:", e)

# Register cleanup function
atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n Shutting down server...')
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def mjpeg_generator():
    """Generator that MJPEG frames with ArUco marker detection overlay and recording capability."""
    global recording, video_writer
    
    while True:
        try:
            frame = get_lastest_frame()
            if frame is None:
                placeholder = cv2.imencode('.jpg', np.zeros((1,1,3), np.uint8))[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                time.sleep(0.1)
                continue
            
            nparr = np.frombuffer(frame, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Process ArUco markers on the BGR frame
                processed_frame = process_aruco_on_frame(cv_image)
                
                # Ghi frame vào video nếu đang recording
                with recording_lock:
                    if recording and video_writer is not None:
                        try:
                            # Resize frame để đảm bảo kích thước phù hợp
                            resized_frame = cv2.resize(processed_frame, (1280, 720))
                            video_writer.write(resized_frame)
                        except Exception as e:
                            print(f"Error writing frame to video: {e}")
                
                # Encode the RGB frame to JPEG
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 80,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ])
                if ret:
                    frame = jpeg.tobytes()
        
        except Exception as e:
            print(f" Error processing frame for ArUco: {e}")
            pass
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

def process_aruco_on_frame(cv_image):
    try:
        global aruco_markers
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # Use legacy API to avoid segfault
        parameters = cv2.aruco.DetectorParameters_create()  # Use legacy API to avoid segfault
        
        # Calculate camera center with offset
        center_x = cv_image.shape[1] // 2 + zone_center_offset[0]
        center_y = cv_image.shape[0] // 2 + zone_center_offset[1]
        
        # Calculate rectangle bounds
        left = center_x - zone_width // 2
        right = center_x + zone_width // 2
        top = center_y - zone_height // 2
        bottom = center_y + zone_height // 2
        
        # Draw zone rectangle and center
        cv2.rectangle(cv_image, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle
        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)  # Center dot
        
        corners, ids, rejected = cv2.aruco.detectMarkers(blur, aruco_dict, parameters=parameters)
        
        detected_count = 0
        if ids is not None:
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id[0])
                corner = corners[i][0]
                marker_center_x = int(np.mean(corner[:, 0]))
                marker_center_y = int(np.mean(corner[:, 1]))
                
                # Check if within rectangle zone
                if left <= marker_center_x <= right and top <= marker_center_y <= bottom:
                    cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                    cv2.putText(cv_image, f"ID: {marker_id}", 
                               (marker_center_x - 90, marker_center_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                    
                    # Draw line from marker center to camera center
                    cv2.line(cv_image, (marker_center_x, marker_center_y), (center_x, center_y), (255, 0, 0), 2)
                    
                    detected_count += 1
        
        cv2.putText(cv_image, f"ArUco Markers Detected: {detected_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 255), 2)
        
    except Exception as e:
        print(f" Error in ArUco frame processing: {e}")
    
    return cv_image

# --- ROUTES ---
@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_aruco_detection', methods=['POST'])
def start_aruco_detection():
    try:
        if controller:
            controller.start_aruco_processing()
            return jsonify({'status': 'success', 'message': 'ArUco detection started'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_aruco_detection', methods=['POST'])
def stop_aruco_detection():
    try:
        if controller:
            controller.stop_aruco_processing()
            return jsonify({'status': 'success', 'message': 'ArUco detection stopped'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, video_writer, recording_start_time
    
    with recording_lock:
        if not recording:
            try:
                # Tạo thư mục recordings nếu chưa tồn tại
                if not os.path.exists('recordings'):
                    os.makedirs('recordings')
                
                # Tạo filename với timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/flight_{timestamp}.avi"
                
                # Khởi tạo video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
                
                recording = True
                recording_start_time = time.time()
                
                print(f"Started recording: {filename}")
                socketio.emit('recording_status', {'status': 'started', 'filename': filename})
                
                return jsonify({'status': 'success', 'message': 'Recording started', 'filename': filename})
            except Exception as e:
                print(f"Error starting recording: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Recording already in progress'}), 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, video_writer, recording_start_time
    
    with recording_lock:
        if recording and video_writer:
            try:
                recording = False
                video_writer.release()
                video_writer = None
                
                duration = time.time() - recording_start_time
                recording_start_time = None
                
                print(f"Stopped recording, duration: {duration:.2f}s")
                socketio.emit('recording_status', {'status': 'stopped', 'duration': duration})
                
                return jsonify({'status': 'success', 'message': 'Recording stopped', 'duration': duration})
            except Exception as e:
                print(f"Error stopping recording: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'No recording in progress'}), 400

@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    try:
        if not os.path.exists('recordings'):
            return jsonify({'recordings': []})
        
        recordings = []
        for filename in os.listdir('recordings'):
            if filename.endswith('.avi'):
                filepath = os.path.join('recordings', filename)
                file_size = os.path.getsize(filepath)
                recordings.append({
                    'filename': filename,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
        
        return jsonify({'recordings': recordings})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/recordings/<filename>')
# def download_recording(filename):
#     try:
#         return send_from_directory('recordings', filename, as_attachment=True)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 404
    
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'drone_connected': controller is not None and controller.is_connected()
    })
@app.route('/update_aruco_markers', methods=['POST'])
def update_aruco_markers():
    """Endpoint để nhận ArUco markers từ drone"""
    global aruco_markers
    try:
        data = request.get_json(silent=True) or {}
        markers = data.get('markers', {})
        
        aruco_markers.update(markers)
        
        socketio.emit('aruco_markers_update', {'markers': aruco_markers})
        
        print(f"Updated ArUco markers: {list(markers.keys())}")
        return jsonify({'status': 'success', 'markers_received': len(markers)})
    except Exception as e:
        print(f"Error updating ArUco markers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_aruco_markers', methods=['GET'])
def get_aruco_markers():
    """Endpoint để lấy danh sách ArUco markers hiện tại"""
    return jsonify(aruco_markers)


@app.route('/set_aruco_ids', methods=['POST'])
def set_aruco_ids():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get('ids', [])
        if not isinstance(ids, list) or not all(isinstance(id, int) for id in ids):
            return jsonify({'error': 'ArUco IDs must be a list of integers'}), 400
        
        if controller:
            controller.set_find_aruco(ids)
            print(f"Updated find_aruco to: {ids}")
            return jsonify({'status': 'success', 'ids': ids})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        print(f" Error setting ArUco IDs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_gps_stations', methods=['GET'])
def get_gps_stations():
    try:
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_mission', methods=['POST'])
def start_mission():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.get_json(silent=True) or {}
        station_name = payload.get('station', 'station1')
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)

        if station_name not in data:
            return jsonify({'error': f'station "{station_name}" not found in JSON'}), 400

        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        waypoints = [start]
        for point in data[station_name]:
            waypoints.append([point['lat'], point['lon']])
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'station': station_name, 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fly_selected', methods=['POST'])
def fly_selected():
    global distance_traveled
    distance_traveled = 0.0
    
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        payload = request.get_json(silent=True) or {}
        points = payload.get('points', [])
        
        if not points:
            return jsonify({'error': 'No points selected'}), 400
            
        v = controller.vehicle
        start = [v.location.global_frame.lat, v.location.global_frame.lon]
        waypoints = [start]
        for point in points:
            waypoints.append([point['lat'], point['lon']])
        
        socketio.emit('planned_path', {'waypoints': waypoints})
        run_mission_in_thread(waypoints)
        return jsonify({'status': 'mission_started', 'waypoints': waypoints})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/return_home', methods=['POST'])
def return_home():
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
        v = controller.vehicle
        home_lat = v.location.global_frame.lat
        home_lon = v.location.global_frame.lon
        home_alt = 3
        home_location = LocationGlobalRelative(home_lat, home_lon, home_alt)
        v.simple_goto(home_location)
        return jsonify({'status': 'returning_home', 'home': [home_lat, home_lon]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_mission_in_thread(waypoints):
    def mission():
        try:
            socketio.emit('mission_status', {'status': 'starting', 'waypoints': waypoints})
            if controller:
                controller.fly_and_precision_land_with_waypoints(waypoints, loiter_alt=3, aruco_duration=60)
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
    try:
        if not controller:
            return jsonify({'error': 'Controller not available'}), 500
            
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('change_mode')
def handle_change_mode(data):
    mode = data.get('mode')
    if mode in ['LAND', 'GUIDED']:
        try:
            if controller:
                controller.vehicle.mode = VehicleMode(mode)
                emit('mission_status', {'status': f'mode_changed_to_{mode}'})
            else:
                emit('mission_status', {'status': 'error', 'error': 'Controller not available'})
        except Exception as e:
            emit('mission_status', {'status': 'error', 'error': str(e)})
    else:
        emit('mission_status', {'status': 'invalid_mode'})

@app.route('/arm', methods=['POST'])
def arm_drone():
    try:
        if controller:
            success = controller.arm_drone()
            if success:
                return jsonify({'status': 'armed', 'message': 'Drone armed successfully'})
            else:
                return jsonify({'error': 'Failed to arm drone'}), 500
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def telemetry_loop():
    """Send telemetry data to clients periodically"""
    while True:
        try:
            if controller and controller.vehicle:
                v = controller.vehicle
                data = {
                    'lat': v.location.global_frame.lat,
                    'lon': v.location.global_frame.lon,
                    'alt': v.location.global_relative_frame.alt,
                    'mode': v.mode.name,
                    'velocity': math.sqrt(v.velocity[0]**2 + v.velocity[1]**2 + v.velocity[2]**2),
                    'distance_traveled': distance_traveled
                }
                socketio.emit('telemetry', data)
            time.sleep(0.5)
        except Exception as e:
            print(f" Telemetry error: {e}")
            time.sleep(1)

if __name__ == '__main__':
    print("Starting Drone Control Server...")
    
    hostname, main_ip, all_ips = get_network_info()
    
    print("🔍 Network Information:")
    print(f"   Hostname: {hostname}")
    print(f"   Main IP: {main_ip}")
    print(f"   All IPs: {', '.join(all_ips)}")
    
    if check_port_availability(5000):
        print("Port 5000 is available")
    else:
        print("Port 5000 might be in use, trying anyway...")
    
    print("Starting telemetry thread...")
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    
    # print("=" * 70)
    # print("DRONE CONTROL SERVER - ACCESS INSTRUCTIONS")
    # print("=" * 70)
    # print(f"On This Machine (Raspberry Pi):")
    # print(f"   → http://localhost:5000")
    # print(f"   → http://127.0.0.1:5000")
    # print()
    # print(f"🌐 From Other Devices (Same Network):")
    # for ip in all_ips:
    #     print(f"   → http://{ip}:5000")
    # print()
    # print(f"Live Video Feed:")
    # print(f"   → http://{main_ip}:5000/video_feed")
    # print()
    # print(f"Health Check:")
    # print(f"   → http://{main_ip}:5000/health")
    # print("=" * 70)
    # print("💡 Tip: Use the main IP address from another device")
    # print("=" * 70)
    
    try:
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        cleanup()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup()