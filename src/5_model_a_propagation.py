# filename: model_a_propagation.py
# description: (Refactored) Model A: Line-Source Propagation.
#              Calculates inland influence based on distance decay from the nearest coastline point.

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import xarray as xr
from scipy.interpolate import griddata
import rasterio

# --- 1. 引入路径配置 ---
# 注意：现在脚本在 src/models/ 下，需要往上跳两级才能找到 src/config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 参数配置 ---
WAVE_VARIABLE = 'VHM0'
ONSHORE_DECAY_FACTOR = 0.02
MAX_INLAND_DISTANCE = 50

# 区域映射 (统一管理)
REGIONS_INFO = {
    "Japan": {'nc': 'japan.nc', 'tif': 'japan_mtbmap.tif'},
    "New_York": {'nc': 'new_york.nc', 'tif': 'new_york_mtbmap.tif'},
    "Melbourne": {'nc': 'melbourne.nc', 'tif': 'melbourne_mtbmap.tif'},
    "Guangzhou_Bay_Area": {'nc': 'guangzhou_bay.nc', 'tif': 'guangzhou_bay_mtbmap.tif'},
    "Hangzhou_Bay": {'nc': 'hangzhou_bay.nc', 'tif': 'hangzhou_bay_mtbmap.tif'}
}

# --- 3. 辅助函数 (HELPER FUNCTIONS - UNTOUCHED) ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return (img > 128).astype(np.uint8)

def save_geotiff(array, ref_path, out_path, nodata_val=0.0):
    try:
        with rasterio.open(ref_path) as ref: profile = ref.profile
        profile.update(dtype=array.dtype.name, count=1, nodata=nodata_val)
        with rasterio.open(out_path, 'w', **profile) as dst: dst.write(array, 1)
        # print(f"GeoTIFF saved: {out_path}")
    except Exception as e: print(f"Error saving GeoTIFF: {e}")

def save_visual_png(data, out_path, title, cmap='viridis'):
    # --- 绘图配置升级 (大字号) ---
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['figure.titlesize'] = 18

    # 宽高比自适应
    height, width = data.shape
    aspect_ratio = height / width
    
    if aspect_ratio > 1.5:  figsize = (8, 12); shrink = 0.5
    elif aspect_ratio < 0.67: figsize = (12, 8); shrink = 0.8
    else: figsize = (10, 10); shrink = 0.7
    
    plt.figure(figsize=figsize, dpi=150)
    
    valid_vals = data[data > 0]
    if valid_vals.size == 0:
        plt.imshow(np.zeros_like(data), cmap='gray'); plt.title(title)
    else:
        vmin = np.percentile(valid_vals, 2)
        vmax = np.percentile(valid_vals, 98)
        current_cmap = plt.get_cmap(cmap).copy(); current_cmap.set_bad(color='lightgray')
        plt.imshow(np.where(data > 0, data, np.nan), cmap=current_cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=title.split(' - ')[0], shrink=shrink, pad=0.02)
        plt.title(title)
    
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Visual saved: {os.path.basename(out_path)}")

# --- 4. 核心算法 (CORE ALGORITHMS - UNTOUCHED) ---

def calculate_mean_field(wave_ds, wave_var):
    """Calculates ONLY the mean field from the full time-series."""
    # print(f"   Calculating mean for '{wave_var}'...")
    h_mean = wave_ds[wave_var].mean(dim='time', skipna=True)
    return h_mean

def interpolate_dataarray_to_raster(data_array, target_shape):
    # This is your trusted interpolation logic
    source_lat = data_array['latitude'].values; source_lon = data_array['longitude'].values
    source_vals = data_array.values
    
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    valid_mask = ~np.isnan(source_vals)
    
    if not np.any(valid_mask): return np.zeros(target_shape, dtype=np.float32)
    
    points = np.vstack((source_lat_grid[valid_mask], source_lon_grid[valid_mask])).T
    values = source_vals[valid_mask]
    
    target_lat_vec = np.linspace(source_lat.max(), source_lat.min(), target_shape[0])
    target_lon_vec = np.linspace(source_lon.min(), source_lon.max(), target_shape[1])
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon_vec, target_lat_vec)
    
    interp_values = griddata(points, values, (target_lat_grid, target_lon_grid), method='nearest')
    return interp_values

def ensure_coastline_values(coast_map, coast_mask):
    # Your trusted function to fill gaps
    valid_coast = (np.nan_to_num(coast_map) > 1e-6)
    invalid_coast = (coast_mask == 1) & (~valid_coast)
    
    if not np.any(invalid_coast) or not np.any(valid_coast): return coast_map
    
    _, indices = distance_transform_edt(~valid_coast, return_indices=True)
    invalid_r, invalid_c = np.where(invalid_coast)
    source_r, source_c = indices[0, invalid_r, invalid_c], indices[1, invalid_r, invalid_c]
    
    filled_map = coast_map.copy()
    filled_map[invalid_r, invalid_c] = filled_map[source_r, source_c]
    return filled_map

def propagate_waves_inland(land_mask, coast_wave_map, decay, max_dist):
    # Your trusted propagation function
    source_mask = coast_wave_map > 0
    if not np.any(source_mask): return np.zeros_like(land_mask, dtype=np.float32)
    
    dist, indices = distance_transform_edt(~source_mask, return_indices=True)
    inland_map = np.zeros_like(land_mask, dtype=np.float32)
    
    onshore_mask = land_mask.astype(bool) & ~source_mask & (dist <= max_dist)
    onshore_r, onshore_c = np.where(onshore_mask)
    
    if onshore_r.size == 0:
        inland_map[source_mask] = coast_wave_map[source_mask]
        return inland_map
        
    source_r, source_c = indices[0, onshore_r, onshore_c], indices[1, onshore_r, onshore_c]
    source_heights = coast_wave_map[source_r, source_c]
    distances = dist[onshore_r, onshore_c]
    
    inland_influence = source_heights * np.exp(-decay * distances)
    
    inland_map[onshore_r, onshore_c] = inland_influence
    inland_map[source_mask] = coast_wave_map[source_mask] # Keep coast values
    return inland_map

# --- 5. 区域处理流程 ---

def process_region(region_name):
    # 路径构造
    file_info = REGIONS_INFO[region_name]
    
    # Inputs (from Step 1 & 2)
    analysis_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
    # NC Input
    nc_path = os.path.join(config.NC_DIR, file_info['nc'])
    # Ref Tiff
    ref_tiff_path = os.path.join(config.TIFF_DIR, file_info['tif'])
    
    # Output Dir (Dedicated for Model A)
    # output_dir = os.path.join(config.OUTPUT_ROOT, 'inland_wave_influence_output_mean_only', region_name)
    # 简化一点，还是放在 outputs/model_a_results/RegionName 下
    output_dir = os.path.join(config.OUTPUT_ROOT, 'model_a_results', region_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"\n--- Model A Processing: {region_name} ---")

    try:
        # Load Geometry
        land_mask = load_image(os.path.join(analysis_dir, 'binary_land_mask.png'))
        coast_mask = load_image(os.path.join(analysis_dir, 'raster_coastline.png'))
        
        # Load Wave Data
        wave_ds = xr.open_dataset(nc_path)
        
        # 1. Calc Mean
        print("   Calculating H_mean field...")
        mean_field_low_res = calculate_mean_field(wave_ds, WAVE_VARIABLE)
        
        # 2. Interpolate
        print("   Interpolating to grid...")
        interp_mean = interpolate_dataarray_to_raster(mean_field_low_res, land_mask.shape)
        
        # 3. Assign to Coastline
        print("   Assigning to coastline...")
        initial_coast_map = np.zeros_like(land_mask, dtype=np.float32)
        initial_coast_map[coast_mask == 1] = interp_mean[coast_mask == 1]
        coast_wave_map = ensure_coastline_values(initial_coast_map, coast_mask)
        
        # 4. Propagate
        print(f"   Propagating inland...")
        inland_map = propagate_waves_inland(land_mask, coast_wave_map, ONSHORE_DECAY_FACTOR, MAX_INLAND_DISTANCE)

        # 5. Save
        output_basename = 'inland_influence_mean_only'
        # PNG
        save_visual_png(inland_map, os.path.join(output_dir, f'{output_basename}.png'), f'Model A: Line Source - {region_name}')
        # GeoTiff (Crucial for later fusion)
        save_geotiff(inland_map, ref_tiff_path, os.path.join(output_dir, f'{output_basename}.tif'))
        # NPY (Optional, fast read)
        np.save(os.path.join(output_dir, f'{output_basename}.npy'), inland_map)

        wave_ds.close()
        print(f"✅ Model A finished for {region_name}")

    except Exception as e:
        print(f"❌ FATAL ERROR for {region_name}: {e}")
        import traceback
        traceback.print_exc()

# --- 6. 主程序 ---
def main():
    print(f"\n🚀 Step 5: Running Model A (Line-Source Propagation)...")
    
    # 确保 REGIONS_INFO 里的名字和 Step 1 生成的文件夹名字一致
    for region_name in REGIONS_INFO.keys():
        process_region(region_name)
        
    print(f"\n✨ Step 5 Completed.")

if __name__ == '__main__':
    main()