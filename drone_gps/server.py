# server.py (updated to support selected points)
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading, time
from drone_control import get_controller
import json
from planner import run_planner  # Import planner
from dronekit import VehicleMode
import math
from dronekit import LocationGlobalRelative

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create/connect controller singleton
controller = get_controller(connection_str='tcp:127.0.0.1:5763', takeoff_height=3)

# Theo dõi distance traveled (tích lũy)
prev_loc = None
distance_traveled = 0.0

def telemetry_loop():
    global prev_loc, distance_traveled
    while True:
        try:
            v = controller.vehicle
            loc = v.location.global_frame
            if loc.lat is None or loc.lon is None:
                time.sleep(0.5)
                continue
            current_loc = (loc.lat, loc.lon)
            velocity = v.groundspeed if v.groundspeed else 0.0
            if prev_loc:
                dlat = current_loc[0] - prev_loc[0]
                dlon = current_loc[1] - prev_loc[1]
                dist_step = math.sqrt((dlat**2) + (dlon**2)) * 1.113195e5  # meters
                distance_traveled += dist_step
            prev_loc = current_loc
            data = {
                'lat': loc.lat,
                'lon': loc.lon,
                'alt': v.location.global_relative_frame.alt,
                'mode': str(v.mode.name),
                'velocity': velocity,
                'distance_traveled': distance_traveled
            }
            socketio.emit('telemetry', data)
        except Exception as e:
            socketio.emit('telemetry', {'error': str(e)})
        time.sleep(0.5)

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

if __name__ == '__main__':
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000)