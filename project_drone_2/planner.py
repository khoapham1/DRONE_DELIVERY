# planner.py (no changes needed for your request, but ensure elevation.tif is generated correctly with the fix above to avoid fallback if desired)
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import heapq
from rasterio.transform import xy, rowcol
from pyproj import Transformer
import numpy as np
import json

# Đường dẫn đến file elevation TIF (thay đổi nếu cần)
ELEVATION_TIF = 'project_drone_2/elevation_ute.tif'  # Giả sử file này tồn tại

def load_and_reproject(tif_path, target_crs='EPSG:32633'):
    with rasterio.open(tif_path) as src:
        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
        dst = np.zeros((height, width), dtype=src.dtypes[0])
        reproject(source=rasterio.band(src, 1), destination=dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=target_crs, resampling=Resampling.bilinear)
    return dst, transform, kwargs

def make_costmap(elev_array):
    # simple: cost = 1 + normalized elevation
    a = elev_array.astype(float)
    a = np.nan_to_num(a, nan=0.0)
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-6)
    cost = 1.0 + 9.0 * a_norm  # cost in [1,10]
    # example: set very steep places to large cost (you can compute slope)
    return cost

# A* on 2D grid (4- or 8-neighbor)
######## VARIABLES ########
def astar(costmap, start_idx, goal_idx, allow_diagonal=True):
    h = lambda a, b: np.hypot(a[0] - b[0], a[1] - b[1])
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    H, W = costmap.shape
    openq = []
    gscore = {start_idx: 0}
    fscore = {start_idx: h(start_idx, goal_idx)}
    heapq.heappush(openq, (fscore[start_idx], start_idx))
    came_from = {}
    while openq:
        _, current = heapq.heappop(openq)
        if current == goal_idx:
            # reconstruct
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            return path[::-1]
        for d in neighbors:
            nb = (current[0] + d[0], current[1] + d[1])
            if nb[0] < 0 or nb[0] >= H or nb[1] < 0 or nb[1] >= W: continue
            tentative_g = gscore[current] + 0.5 * (costmap[current] + costmap[nb]) * (np.hypot(d[0], d[1]))
            if tentative_g < gscore.get(nb, 1e18):
                came_from[nb] = current
                gscore[nb] = tentative_g
                fscore[nb] = tentative_g + h(nb, goal_idx)
                heapq.heappush(openq, (fscore[nb], nb))
    return None  # no path

def run_planner(payload):
    start = payload["start"]  # [lat, lon]
    goal = payload["goal"]    # [lat, lon]

    try:
        # Load elevation và tạo costmap
        elev_array, transform, kwargs = load_and_reproject(ELEVATION_TIF)
        costmap = make_costmap(elev_array)

        # Chuyển lat/lon sang UTM (x,y) rồi sang grid row/col
        transformer_to_utm = Transformer.from_crs('EPSG:4326', kwargs['crs'], always_xy=True)
        start_utm_x, start_utm_y = transformer_to_utm.transform(start[1], start[0])  # lon, lat
        goal_utm_x, goal_utm_y = transformer_to_utm.transform(goal[1], goal[0])
        start_idx = rowcol(transform, start_utm_x, start_utm_y)
        goal_idx = rowcol(transform, goal_utm_x, goal_utm_y)

        # Chạy A*
        path_indices = astar(costmap, start_idx, goal_idx)
        if path_indices is None:
            raise ValueError("No path found")

        # Chuyển indices ngược về lat/lon
        transformer_to_wgs = Transformer.from_crs(kwargs['crs'], 'EPSG:4326', always_xy=True)
        waypoints = []
        for idx in path_indices[::10]:  # Downsample để tránh quá nhiều points (tối ưu)
            x, y = xy(transform, idx[0], idx[1])
            lon, lat = transformer_to_wgs.transform(x, y)
            waypoints.append([lat, lon])
        return waypoints
    except Exception as e:
        print(f"Planner error: {e}, falling back to GPS stations from JSON")
        try:
            with open('file_gps_station.json', 'r') as f:
                data = json.load(f)
            station_name = payload.get('station', 'station1')
            stations = data.get(station_name, data.get('station1', []))
            waypoints = [[point['lat'], point['lon']] for point in stations]
            waypoints = [payload['start']] + waypoints + [payload['goal']]
            return waypoints
        except Exception as e2:
            print("Fallback error:", e2)
            return [payload['start'], payload['goal']]
