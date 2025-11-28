# filename: 2_analyze_geometry.py
# description: (Refactored) Reads geometry files from output folder,
#              runs buffer and propagation analysis.

import cv2
import numpy as np
import os
import rasterio
from scipy.ndimage import distance_transform_edt, binary_dilation
import matplotlib.pyplot as plt
import sys

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# --- 2. 参数配置 ---
NUM_OFFSHORE_LAYERS = 30
NUM_ONSHORE_LAYERS = 15
BUFFER_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

# 区域配置 (基于 Step 1 的输出结构)
# 只需要定义 ref_tiff_name，路径通过 config 自动构建
REGIONS_INFO = {
    "Japan": 'japan_mtbmap.tif',
    "New_York": 'new_york_mtbmap.tif',
    "Melbourne": 'melbourne_mtbmap.tif',
    "Guangzhou_Bay_Area": 'guangzhou_bay_mtbmap.tif',
    "Hangzhou_Bay": 'hangzhou_bay_mtbmap.tif'
}

# --- 3. 辅助函数 ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return (img > 128).astype(np.uint8)

def load_npy(path):
    if not os.path.exists(path): return None
    return np.load(path)

def create_unified_buffer_level_map(mask, coast, num_off, num_on, kernel):
    # 核心算法：生成离岸和在岸的缓冲层级
    levels = np.zeros(mask.shape, dtype=np.int16)
    sea, land, coastline = ~mask.astype(bool), mask.astype(bool), coast.astype(bool)
    
    # Offshore (向海一侧扩张)
    print("   Generating offshore buffer levels...")
    frontier = coastline.copy(); visited = coastline.copy()
    for i in range(1, num_off + 1):
        new_layer = binary_dilation(frontier, structure=kernel) & sea & ~visited
        if not np.any(new_layer): break
        levels[new_layer] = i; visited |= new_layer; frontier = new_layer
    
    # Onshore (向陆一侧扩张)
    print("   Generating onshore buffer levels...")
    frontier = coastline.copy(); visited = coastline.copy()
    for i in range(1, num_on + 1):
        new_layer = binary_dilation(frontier, structure=kernel) & land & ~visited
        if not np.any(new_layer): break
        levels[new_layer] = -i; visited |= new_layer; frontier = new_layer
        
    return levels

def propagate_coastline_attribute(levels, coast_attr, coast):
    # 核心算法：将海岸线属性（如角度、曲率）传播到最近的海域像素
    if coast_attr is None:
        return None
    
    offshore_mask = levels > 0
    if not np.any(offshore_mask):
        return np.full(levels.shape, np.nan, dtype=np.float32)
    
    # 使用欧几里得距离变换找到最近的海岸点索引
    _, indices = distance_transform_edt(~coast.astype(bool), return_indices=True)
    
    offshore_r, offshore_c = np.where(offshore_mask)
    nearest_r, nearest_c = indices[0, offshore_r, offshore_c], indices[1, offshore_r, offshore_c]
    
    propagated_map = np.full(levels.shape, np.nan, dtype=np.float32)
    propagated_map[offshore_r, offshore_c] = coast_attr[nearest_r, nearest_c]
    
    print(f"   Propagated attribute to {np.sum(offshore_mask)} pixels.")
    return propagated_map
    
def save_geotiff(array, ref_path, out_path, nodata_val=None):
    if array is None: return
    try:
        with rasterio.open(ref_path) as ref: profile = ref.profile
        profile.update(dtype=array.dtype.name, count=1, nodata=nodata_val)
        with rasterio.open(out_path, 'w', **profile) as dst: dst.write(array, 1)
        # print(f"GeoTIFF saved: {out_path}")
    except Exception as e:
        print(f"❌ Error saving GeoTIFF {out_path}: {e}")

def save_visual_png(data_map, output_path, cmap='viridis', title=''):
    if data_map is None: return
    
    # --- 绘图配置升级 (大字号) ---
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # 根据长宽比动态调整画布
    h, w = data_map.shape
    aspect = h / w if w != 0 else 1.0
    plt.figure(figsize=(10, 10 * aspect), dpi=150) # 提高 DPI
    
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color='lightgray') 
    
    plt.imshow(data_map, cmap=current_cmap)
    plt.colorbar(label=title, pad=0.02)
    plt.title(title)
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Visual saved: {os.path.basename(output_path)}")

# --- 4. 区域处理流程 ---

def process_region(region_name, ref_tiff_filename):
    # 路径全部基于 config 动态构建
    # Step 1 的输出就在这里
    analysis_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
    
    if not os.path.exists(analysis_dir):
        print(f"⚠️ SKIPPING: Directory not found -> {analysis_dir}")
        return
        
    print(f"   Analyzing geometry in -> {analysis_dir}")

    try:
        # Load inputs from Step 1
        mask = load_image(os.path.join(analysis_dir, 'binary_land_mask.png'))
        coast = load_image(os.path.join(analysis_dir, 'raster_coastline.png'))
        direction = load_npy(os.path.join(analysis_dir, 'coastline_direction_raw.npy'))
        curvature = load_npy(os.path.join(analysis_dir, 'coastline_curvature_raw.npy'))
        
        # Ref TIFF 也在这个目录里 (Step 1 复制过去的)
        ref_tiff_path = os.path.join(analysis_dir, ref_tiff_filename)
        
        if not os.path.exists(ref_tiff_path):
             # 尝试从源目录找
             ref_tiff_path = os.path.join(config.TIFF_DIR, ref_tiff_filename)

    except FileNotFoundError as e:
        print(f"❌ Error: Missing input file. {e}")
        return

    # Task A: Buffer Levels
    levels = create_unified_buffer_level_map(mask, coast, NUM_OFFSHORE_LAYERS, NUM_ONSHORE_LAYERS, BUFFER_KERNEL)
    np.save(os.path.join(analysis_dir, 'unified_buffer_level_map_raw.npy'), levels)
    save_geotiff(levels, ref_tiff_path, os.path.join(analysis_dir, 'unified_buffer_level_map.tif'), nodata_val=np.iinfo(np.int16).min)
    save_visual_png(levels, os.path.join(analysis_dir, 'unified_buffer_level_map.png'), cmap='coolwarm_r', title='Buffer Levels')

    # Task B: Propagate Direction
    prop_direction = propagate_coastline_attribute(levels, direction, coast)
    if prop_direction is not None:
        np.save(os.path.join(analysis_dir, 'propagated_offshore_direction_raw.npy'), prop_direction)
        save_visual_png(prop_direction, os.path.join(analysis_dir, 'propagated_offshore_direction.png'), cmap='hsv', title='Propagated Direction')
    
    # Task C: Propagate Curvature
    prop_curvature = propagate_coastline_attribute(levels, curvature, coast)
    if prop_curvature is not None:
        np.save(os.path.join(analysis_dir, 'propagated_offshore_curvature_raw.npy'), prop_curvature)
        save_visual_png(prop_curvature, os.path.join(analysis_dir, 'propagated_offshore_curvature.png'), cmap='viridis', title='Propagated Curvature')
    
    print(f"✅ Region {region_name} processed.")

# --- 5. 主程序 ---

def main():
    print(f"\n🚀 Step 2: Geometry Dynamics Analysis...")
    
    for region_name, ref_tiff_name in REGIONS_INFO.items():
        print(f"\n--- Region: {region_name} ---")
        process_region(region_name, ref_tiff_name)
    
    print(f"\n✨ Step 2 Completed.")

if __name__ == '__main__':
    main()