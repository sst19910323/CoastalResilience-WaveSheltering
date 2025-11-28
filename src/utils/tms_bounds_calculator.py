# filename: tms_bounds_calculator.py
# description: Utility to calculate exact Lat/Lon bounds for TMS tile ranges.
#              Useful for defining the bounding box when downloading NetCDF data.

import mercantile
import pyproj

# Regions with their x and y ranges at zoom level 7
# (Same definitions as used in mosaic/download scripts)
REGIONS = {
    "japan": {"x_range": (112, 113), "y_range": (50, 50)},
    "hangzhou_bay": {"x_range": (106, 107), "y_range": (52, 52)},
    "guangzhou_bay": {"x_range": (104, 104), "y_range": (55, 56)},
    "melbourne": {"x_range": (115, 115), "y_range": (78, 78)},
    "new_york": {"x_range": (37, 38), "y_range": (48, 48)}
}

# Zoom level
ZOOM = 7

def get_region_bounds(x_range, y_range, zoom):
    """使用mercantile库获取区域边界"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 获取西北角瓦片的边界
    nw_bounds = mercantile.bounds(x_min, y_min, zoom)
    
    # 获取东南角瓦片的边界
    se_bounds = mercantile.bounds(x_max, y_max, zoom)
    
    # 提取边界坐标 (West, South, East, North)
    return nw_bounds.north, se_bounds.south, nw_bounds.west, se_bounds.east

def calculate_distance(north, south, west, east):
    """计算区域的宽度和高度（公里）"""
    # 使用pyproj计算大圆距离
    geod = pyproj.Geod(ellps='WGS84')
    
    # 计算高度（南北方向）
    _, _, height = geod.inv(west, north, west, south)
    
    # 计算平均纬度上的宽度（东西方向）
    avg_lat = (north + south) / 2
    _, _, width = geod.inv(west, avg_lat, east, avg_lat)
    
    return width / 1000, height / 1000

def convert_to_0_360(longitude):
    """将-180到180度的经度转换为0到360度 (CMEMS格式)"""
    if longitude < 0:
        return longitude + 360
    return longitude

def main():
    print("Region Bounds Information (Zoom Level 7):")
    print("-" * 120)
    print(f"{'Region':<15} {'North':<12} {'South':<12} {'Lon (-180/180)':<30} {'Lon (0/360 - CMEMS)':<30} {'Size (km)'}")
    print("-" * 120)
    
    for region_name, coordinates in REGIONS.items():
        x_range = coordinates["x_range"]
        y_range = coordinates["y_range"]
        
        north, south, west, east = get_region_bounds(x_range, y_range, ZOOM)
        width_km, height_km = calculate_distance(north, south, west, east)
        
        # 转换为0-360度的经度表示
        west_360 = convert_to_0_360(west)
        east_360 = convert_to_0_360(east)
        
        # 格式化经度范围字符串
        lon_range_180 = f"{west:.4f} to {east:.4f}"
        
        # CMEMS通常要求小值在前，大值在后。对于跨越0度或180度的情况需要特殊处理
        # 这里做简单处理用于显示
        if west_360 > east_360:
             # 跨越 0 度 (如 358 to 2)
             lon_range_360 = f"{west_360:.4f} to {east_360:.4f} (Cross)"
        else:
             lon_range_360 = f"{west_360:.4f} to {east_360:.4f}"
        
        print(f"{region_name:<15} {north:12.4f} {south:12.4f} {lon_range_180:<30} {lon_range_360:<30} {width_km:.0f}x{height_km:.0f} km")
    
    print("-" * 120)
    print("NOTE:")
    print("1. Use 'Lon (0/360)' coordinates when downloading from Copernicus Marine Service.")
    print("2. Ensure your NetCDF download covers slightly MORE than these bounds to avoid edge effects.")

if __name__ == "__main__":
    main()