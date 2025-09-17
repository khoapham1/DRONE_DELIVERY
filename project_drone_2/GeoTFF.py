# GeoTFF.py (fixed lon/lat assignment)
from osgeo import gdal
from osgeo import osr
import numpy as np

image_size = (201, 201)  # Kích thước grid
lon = np.zeros(image_size, dtype=np.float64)
lat = np.zeros(image_size, dtype=np.float64)
elevation = np.zeros(image_size, dtype=np.float64)  # Thay bằng dữ liệu elevation thực của bạn

# Tạo dữ liệu mẫu (thay bằng data thực: lon, lat, elevation từ list hoặc file CSV)
for x in range(image_size[0]):
    for y in range(image_size[1]):
        lon[y, x] = 106.7712575 + 0.01 * x  # Longitude từ 106.7712575 (fixed)
        lat[y, x] = 10.8507304 + 0.01 * y   # Latitude từ 10.8507304 (fixed)
        elevation[y, x] = np.random.uniform(0, 1000)  # Elevation ngẫu nhiên 0-1000m (thay bằng data thực)

# Thiết lập geotransform
nx, ny = image_size
xmin, ymin, xmax, ymax = lon.min(), lat.min(), lon.max(), lat.max()
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)
geotransform = (xmin, xres, 0, ymax, 0, -yres)

# Tạo file GeoTIFF 1-band (elevation)
dst_ds = gdal.GetDriverByName('GTiff').Create('elevation_ute.tif', ny, nx, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform(geotransform)  # Thiết lập tọa độ
srs = osr.SpatialReference()         # Thiết lập CRS
srs.ImportFromEPSG(4326)             # WGS84 lat/lon
dst_ds.SetProjection(srs.ExportToWkt())  # Export CRS
dst_ds.GetRasterBand(1).WriteArray(elevation)  # Ghi dữ liệu elevation
dst_ds.FlushCache()                  # Lưu file
dst_ds = None                        # Đóng file