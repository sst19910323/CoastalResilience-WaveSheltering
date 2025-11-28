# filename: tms_mosaic_tool.py
# description: Utility to mosaic TMS tiles into georeferenced GeoTIFFs.
#              This creates the base reference maps for the analysis.

import os
import glob
import numpy as np
from osgeo import gdal, osr
from PIL import Image
import sys

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 配置 ---

# 瓦片源配置 (基于 config 路径)
TILE_SOURCES = {
    "mtbmap": {
        "dir": os.path.join(config.TMS_DIR, "mtbmap")
    },
    "lobelia": {
        "dir": os.path.join(config.TMS_DIR, "lobelia")
    }
}

# 输出TIFF目录
OUTPUT_DIR = config.TIFF_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 瓦片尺寸（像素）
TILE_SIZE = 256

# Web Mercator投影的一些常量
EARTH_RADIUS = 6378137.0  # 地球半径（米）
ORIGIN_SHIFT = np.pi * EARTH_RADIUS  # 坐标系原点偏移

# --- 3. 辅助函数 ---

def tile_to_mercator(tx, ty, zoom):
    """将瓦片坐标转换为Web Mercator投影坐标（米）"""
    n = 2.0 ** zoom
    resolution = 2 * ORIGIN_SHIFT / (TILE_SIZE * n)
    x = tx * TILE_SIZE * resolution - ORIGIN_SHIFT
    y = ORIGIN_SHIFT - ty * TILE_SIZE * resolution
    return x, y, resolution

def get_bounds_for_zoom_region(zoom, x_min, x_max, y_min, y_max):
    """计算指定区域的地理范围"""
    min_x, max_y, resolution = tile_to_mercator(x_min, y_min, zoom)
    max_x, min_y, _ = tile_to_mercator(x_max + 1, y_max + 1, zoom)
    width_pixels = (x_max - x_min + 1) * TILE_SIZE
    height_pixels = (y_max - y_min + 1) * TILE_SIZE
    return min_x, max_y, max_x, min_y, width_pixels, height_pixels, resolution

def create_empty_tiff(output_path, width, height, min_x, max_y, resolution):
    """创建空白的地理参考TIFF文件"""
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, 3, gdal.GDT_Byte)
    
    # 设置变换参数 (垂直分辨率为负值)
    dataset.SetGeoTransform((min_x, resolution, 0, max_y, 0, -resolution))
    
    # 设置投影坐标系 (Web Mercator EPSG:3857)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    
    return dataset

def process_region(region_name, zoom, x_min, x_max, y_min, y_max, source_key):
    """处理一个区域的所有瓦片，创建TIFF文件"""
    # 1. 计算范围
    min_x, max_y, max_x, min_y, width, height, resolution = get_bounds_for_zoom_region(
        zoom, x_min, x_max, y_min, y_max
    )
    
    output_path = os.path.join(OUTPUT_DIR, f"{region_name}_{source_key}.tif")
    
    print(f"Creating Mosaic: {region_name} ({source_key})")
    print(f"   Size: {width}x{height}, Res: {resolution:.2f}m")
    
    # 2. 创建空TIFF
    tiff = create_empty_tiff(output_path, width, height, min_x, max_y, resolution)
    data = np.zeros((3, height, width), dtype=np.uint8)
    
    # 3. 填充瓦片
    missing_tiles = 0
    processed_tiles = 0
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = os.path.join(TILE_SOURCES[source_key]["dir"], f"{zoom}_{x}_{y}.png")
            
            if os.path.exists(tile_path):
                processed_tiles += 1
                try:
                    with Image.open(tile_path) as tile_img:
                        if tile_img.mode != 'RGB': tile_img = tile_img.convert('RGB')
                        tile_array = np.array(tile_img)
                    
                    local_x = (x - x_min) * TILE_SIZE
                    local_y = (y - y_min) * TILE_SIZE
                    
                    data[0, local_y:local_y+TILE_SIZE, local_x:local_x+TILE_SIZE] = tile_array[:, :, 0]
                    data[1, local_y:local_y+TILE_SIZE, local_x:local_x+TILE_SIZE] = tile_array[:, :, 1]
                    data[2, local_y:local_y+TILE_SIZE, local_x:local_x+TILE_SIZE] = tile_array[:, :, 2]
                except Exception as e:
                    print(f"   Error reading tile {tile_path}: {e}")
            else:
                missing_tiles += 1
                # print(f"   Missing tile: {tile_path}")
    
    # 4. 写入TIFF
    for band_idx in range(3):
        tiff.GetRasterBand(band_idx + 1).WriteArray(data[band_idx])
    
    tiff = None # Close file
    print(f"   Saved: {output_path}")

# --- 4. 主程序 ---

def main():
    # 定义要处理的区域 (瓦片坐标范围)
    regions = {
        "japan": {"x_range": (112, 113), "y_range": (50, 50)},
        "hangzhou_bay": {"x_range": (106, 107), "y_range": (52, 52)},
        "guangzhou_bay": {"x_range": (104, 104), "y_range": (55, 56)},
        "melbourne": {"x_range": (115, 115), "y_range": (78, 78)},
        "new_york": {"x_range": (37, 38), "y_range": (48, 48)}
    }
    
    zoom = 7
    
    for region_name, coordinates in regions.items():
        for source_key in TILE_SOURCES.keys():
            x_min, x_max = coordinates["x_range"]
            y_min, y_max = coordinates["y_range"]
            process_region(region_name, zoom, x_min, x_max, y_min, y_max, source_key)

if __name__ == "__main__":
    main()