# filename: experimental_fusion_attempt.py
# description: [EXPERIMENTAL] Attempt to fuse Mean Field with the Effective Wave Component.
#              Since the Effective Wave Component was discarded, this fusion approach was also deprecated
#              in favor of the Model A + Model B fusion.

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config

# --- 2. 参数配置 ---
WAVE_VARIABLE = 'VHM0'
ONSHORE_DECAY_FACTOR = 0.02
MAX_INLAND_DISTANCE = 50
COMBINED_WEIGHT_MEAN = 0.5
COMBINED_WEIGHT_EFFECTIVE = 0.5

# 仅在部分区域测试
REGIONS_CONFIG = {
    "Melbourne": {"geometry_folder": "Melbourne"}, 
    "New_York": {"geometry_folder": "New_York"}
}

# --- 3. 辅助函数 ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return (img > 128).astype(np.uint8)

def save_geotiff(array, ref_path, out_path, nodata_val=0.0):
    try:
        with rasterio.open(ref_path) as ref: profile = ref.profile
        profile.update(dtype=array.dtype.name, count=1, nodata=nodata_val)
        with rasterio.open(out_path, 'w', **profile) as dst: dst.write(array, 1)
    except Exception as e: print(f"Error saving GeoTIFF: {e}")

def save_visual_png(data, out_path, title, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    valid_vals = data[data > 0]
    if valid_vals.size == 0:
        plt.imshow(np.zeros_like(data), cmap='gray'); plt.title(title)
    else:
        vmin = np.percentile(valid_vals, 2)
        vmax = np.percentile(valid_vals, 98)
        current_cmap = plt.get_cmap(cmap).copy(); current_cmap.set_bad(color='lightgray')
        plt.imshow(np.where(data > 0, data, np.nan), cmap=current_cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label=title.split(' - ')[0])
        plt.title(title + " (Exp)")
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def interpolate_dataarray_to_raster(data_array, ref_tiff_path):
    with rasterio.open(ref_tiff_path) as ref:
        target_height, target_width = ref.height, ref.width
        target_bounds = ref.bounds
    source_lat = data_array['latitude'].values; source_lon = data_array['longitude'].values
    source_vals = data_array.values
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    points = np.vstack((source_lat_grid.ravel(), source_lon_grid.ravel())).T
    values = source_vals.ravel()
    target_lat_vec = np.linspace(target_bounds.top, target_bounds.bottom, target_height)
    target_lon_vec = np.linspace(target_bounds.left, target_bounds.right, target_width)
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon_vec, target_lat_vec)
    interp_values = griddata(points, values, (target_lat_grid, target_lon_grid), method='nearest')
    return interp_values

def propagate_waves_inland(land_mask, source_wave_map, decay, max_dist):
    source_mask = source_wave_map >= 0
    if not np.any(source_mask):
        return np.zeros_like(land_mask, dtype=np.float32)
        
    dist, indices = distance_transform_edt(~source_mask, return_indices=True)
    inland_map = np.zeros_like(land_mask, dtype=np.float32)
    onshore_mask = land_mask.astype(bool) & (dist <= max_dist)
    onshore_mask[source_mask] = False
    
    onshore_r, onshore_c = np.where(onshore_mask)
    if onshore_r.size == 0: 
        inland_map[source_mask] = source_wave_map[source_mask]
        return inland_map

    source_r, source_c = indices[0, onshore_r, onshore_c], indices[1, onshore_r, onshore_c]
    source_heights = source_wave_map[source_r, source_c]
    distances = dist[onshore_r, onshore_c]
    
    inland_influence = source_heights * np.exp(-decay * distances)
    inland_map[onshore_r, onshore_c] = inland_influence
    inland_map[source_mask] = source_wave_map[source_mask]
    return inland_map

# --- 4. 主流程 ---

def process_region(region_name, cfg):
    print(f"\n--- Processing Experimental Fusion: {region_name} ---")
    
    # 路径构造 (需要手动匹配，因为之前的 config 没有针对 experimental 的路径)
    # 这里假设输入都在 experimental_outputs 下
    exp_input_dir = os.path.join(config.OUTPUT_ROOT, 'experimental_outputs', region_name)
    effective_tif = os.path.join(exp_input_dir, 'VHM0_effective_mean_experimental.tif') # 假设上一步生成了这个
    
    # Geometry & NC
    geometry_folder = cfg["geometry_folder"]
    analysis_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', geometry_folder)
    nc_path = os.path.join(config.NC_DIR, config.REGIONS_CONFIG[region_name]['nc_file'] if region_name in config.REGIONS_CONFIG else 'melbourne.nc') # Hacky
    ref_tiff_path = os.path.join(config.TIFF_DIR, f"{region_name.lower()}_mtbmap.tif")
    if region_name == 'New_York': ref_tiff_path = os.path.join(config.TIFF_DIR, 'new_york_mtbmap.tif')

    output_dir = os.path.join(config.OUTPUT_ROOT, 'experimental_outputs', region_name)

    if not os.path.exists(effective_tif):
        print(f"Skipping {region_name}: Effective wave TIFF not found. Run calculate_effective_wave_component.py first.")
        return

    try:
        land_mask = load_image(os.path.join(analysis_dir, 'binary_land_mask.png'))
        
        with rasterio.open(effective_wave_map) as src:
            effective_wave_map = src.read(1)
            
        wave_ds = xr.open_dataset(nc_path)
        h_mean_low_res = wave_ds[WAVE_VARIABLE].mean(dim='time', skipna=True)
        interp_h_mean = interpolate_dataarray_to_raster(h_mean_low_res, ref_tiff_path)
        
        # Combine
        interp_h_mean_safe = np.where(interp_h_mean >= 0, interp_h_mean, 0)
        effective_wave_map_safe = np.where(effective_wave_map >= 0, effective_wave_map, 0)
        combined_map = (COMBINED_WEIGHT_MEAN * interp_h_mean_safe) + (COMBINED_WEIGHT_EFFECTIVE * effective_wave_map_safe)
        
        # Propagate
        inland_map_combined = propagate_waves_inland(land_mask, combined_map, ONSHORE_DECAY_FACTOR, MAX_INLAND_DISTANCE)
        
        save_visual_png(inland_map_combined, os.path.join(output_dir, 'inland_influence_exp_combined.png'), f'Inland Influence (Exp Combined) - {region_name}')
        save_geotiff(inland_map_combined, ref_tiff_path, os.path.join(output_dir, 'inland_influence_exp_combined.tif'))
        
        wave_ds.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    for r, cfg in REGIONS_CONFIG.items():
        process_region(r, cfg)