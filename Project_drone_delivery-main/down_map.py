import os
import requests
import math

def latlon_to_tile(lat_deg, lon_deg, zoom):
    """Chuyển lat/lon sang tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_tiles(min_zoom, max_zoom, min_lat, max_lat, min_lon, max_lon, tile_server='https://tile.openstreetmap.org/{z}/{x}/{y}.png'):
    """Download tiles cho bounds và zoom levels."""
    for z in range(min_zoom, max_zoom + 1):
        min_x, max_y = latlon_to_tile(max_lat, min_lon, z)  # Top-left
        max_x, min_y = latlon_to_tile(min_lat, max_lon, z)  # Bottom-right
        
        print(f"Downloading zoom {z}: x={min_x}-{max_x}, y={min_y}-{max_y}")
        
        os.makedirs(f'tiles/{z}', exist_ok=True)
        for x in range(min_x, max_x + 1):
            os.makedirs(f'tiles/{z}/{x}', exist_ok=True)
            for y in range(min_y, max_y + 1):
                url = tile_server.format(z=z, x=x, y=y)
                path = f'tiles/{z}/{x}/{y}.png'
                if os.path.exists(path):
                    print(f"Skip {path} (already exists)")
                    continue
                
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        with open(path, 'wb') as f:
                            f.write(resp.content)
                        print(f"Downloaded {path}")
                    else:
                        print(f"Error {resp.status_code} for {url}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

# Params cho khu vực HCMUTE (có thể chỉnh rộng hơn nếu cần)
min_lat = 10.84
max_lat = 10.86
min_lon = 106.76
max_lon = 106.78
min_zoom = 10
max_zoom = 20  # Tăng nếu cần chi tiết hơn, nhưng file sẽ lớn

download_tiles(min_zoom, max_zoom, min_lat, max_lat, min_lon, max_lon)