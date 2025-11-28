# filename: 1_analyze_coastline.py
# description: (Refactored) Reads source GeoTIFFs from 'data/ref_tiffs'
#              and generates binary masks, coastline rasters, direction, and curvature.

import cv2
import numpy as np
import rasterio
import os
import shutil
import sys

# --- 1. 引入路径配置 (CONFIG IMPORT) ---
# 将当前脚本所在目录加入 path，以便导入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# --- 2. 参数配置 (PARAMETERS) ---
# 这里的 RGB 和 Tolerance 是你试出来的黄金参数，原样保留
LAND_RGB = np.array([242, 239, 233]) 
COLOR_TOLERANCE = 5
COASTLINE_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
DIRECTION_WINDOW_SIZE = 3
CURVATURE_ANALYSIS_WINDOW_SIZE = 5

# 区域映射 (文件名 -> 区域文件夹名)
# 键是 config.TIFF_DIR 下的文件名，值是 output 下的子文件夹名
REGIONS_MAP = {
    'japan_mtbmap.tif': 'Japan',
    'new_york_mtbmap.tif': 'New_York',
    'melbourne_mtbmap.tif': 'Melbourne',
    'guangzhou_bay_mtbmap.tif': 'Guangzhou_Bay_Area',
    'hangzhou_bay_mtbmap.tif': 'Hangzhou_Bay'
}

# --- 3. 核心算法 (CORE ALGORITHMS - UNTOUCHED) ---

def create_output_directory(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def rgb_to_binary_land_mask(tiff_path, land_rgb_val, tolerance):
    try:
        with rasterio.open(tiff_path) as src:
            if src.count < 3: return None
            print(f"   Reading TIFF: {os.path.basename(tiff_path)}")
            # 读取前3个波段 (R, G, B)
            rgb = np.stack([src.read(1), src.read(2), src.read(3)], axis=-1)
            # 使用 OpenCV 提取特定颜色的掩膜
            mask = cv2.inRange(rgb, np.clip(land_rgb_val-tolerance,0,255), np.clip(land_rgb_val+tolerance,0,255))
            return (mask / 255).astype(np.uint8)
    except Exception as e: 
        print(f"❌ Error reading {tiff_path}: {e}")
        return None

def extract_raster_coastline_from_mask(binary_land_mask, kernel):
    # 腐蚀法提取边缘：mask - eroded_mask
    if binary_land_mask is None or np.all(binary_land_mask == 0) or np.all(binary_land_mask == 1): return None
    return binary_land_mask - cv2.erode(binary_land_mask, kernel, iterations=1)

def calculate_coastline_direction(binary_land_mask, coastline_raster):
    if coastline_raster is None: return None
    # 使用 Sobel 算子计算 X 和 Y 方向的梯度
    sobel_x = cv2.Sobel(binary_land_mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=DIRECTION_WINDOW_SIZE)
    sobel_y = cv2.Sobel(binary_land_mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=DIRECTION_WINDOW_SIZE)
    # 计算梯度的角度
    angle_deg = np.degrees(cv2.phase(sobel_x, sobel_y, angleInDegrees=False))
    
    # 只保留海岸线像素上的角度
    direction_map = np.full_like(angle_deg, np.nan, dtype=np.float32)
    direction_map[coastline_raster == 1] = angle_deg[coastline_raster == 1]
    return direction_map

def calculate_coastline_curvature(coastline_raster, direction_map, window_size):
    if direction_map is None: return None
    curvature_map = np.full_like(direction_map, np.nan, dtype=np.float32)
    coords = list(zip(*np.where(coastline_raster == 1)))
    
    if not coords: return curvature_map
    
    pad = window_size // 2
    # 填充边缘以处理边界情况
    padded_dirs = np.pad(direction_map, pad, mode='reflect')
    padded_coast = np.pad(coastline_raster, pad, mode='constant')
    
    # 滑动窗口计算曲率 (角度的标准差)
    for r, c in coords:
        win_dirs = padded_dirs[r:r+2*pad+1, c:c+2*pad+1]
        win_coast = padded_coast[r:r+2*pad+1, c:c+2*pad+1]
        
        valid_dirs = win_dirs[win_coast == 1]
        valid_dirs = valid_dirs[~np.isnan(valid_dirs)]
        
        if len(valid_dirs) < 2: continue
        
        # 处理 0/360 度跨越问题
        diffs = (valid_dirs - direction_map[r, c] + 180) % 360 - 180
        curvature_map[r, c] = np.std(diffs)
        
    return curvature_map

# --- 4. 区域处理流程 (PROCESS REGION) ---

def process_region(tiff_filename, region_name):
    input_tiff_path = os.path.join(config.TIFF_DIR, tiff_filename)
    
    # 输出目录改到 config.OUTPUT_ROOT/raster_geometry_analysis_output/region_name
    # 保持你原来的结构习惯
    region_output_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
    
    if not os.path.exists(input_tiff_path):
        print(f"⚠️ SKIPPING: Input file not found -> {input_tiff_path}")
        return
    
    create_output_directory(region_output_dir)
    print(f"   Processing -> {region_output_dir}")

    # 执行核心算法
    mask = rgb_to_binary_land_mask(input_tiff_path, LAND_RGB, COLOR_TOLERANCE)
    
    if mask is None:
        print(f"⚠️ Warning: Failed to generate mask for {region_name}")
        return

    coast = extract_raster_coastline_from_mask(mask, COASTLINE_KERNEL)
    direction = calculate_coastline_direction(mask, coast)
    curvature = calculate_coastline_curvature(coast, direction, CURVATURE_ANALYSIS_WINDOW_SIZE)
    
    # 保存结果
    if mask is not None: cv2.imwrite(os.path.join(region_output_dir, 'binary_land_mask.png'), mask * 255)
    if coast is not None: cv2.imwrite(os.path.join(region_output_dir, 'raster_coastline.png'), coast * 255)
    if direction is not None: np.save(os.path.join(region_output_dir, 'coastline_direction_raw.npy'), direction)
    if curvature is not None: np.save(os.path.join(region_output_dir, 'coastline_curvature_raw.npy'), curvature)
    
    # 关键：复制原始 TIF 到输出目录，因为后续步骤需要它做基准
    shutil.copy(input_tiff_path, os.path.join(region_output_dir, os.path.basename(input_tiff_path)))
    print(f"✅ Saved results for {region_name}")

# --- 5. 主程序 (MAIN) ---

def main():
    print(f"\n🚀 Step 1: Analyzing Raster Coastline Geometry...")
    print(f"   Input Dir: {config.TIFF_DIR}")
    
    for tiff_file, region_name in REGIONS_MAP.items():
        print(f"\n--- Region: {region_name} ---")
        process_region(tiff_file, region_name)
    
    print(f"\n✨ Step 1 Completed.")

if __name__ == '__main__':
    main()