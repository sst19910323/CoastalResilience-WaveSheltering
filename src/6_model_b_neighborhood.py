# filename: 6_model_b_neighborhood.py
# description: (Refactored) Model B: Neighborhood Average Model.
#              Calculates inland influence by averaging wave height in a coastal neighborhood.

import numpy as np
import os
import sys
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
import cv2
import matplotlib.pyplot as plt
import rasterio
import time

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 参数配置 ---
WAVE_VARIABLE = 'VHM0'
NEIGHBORHOOD_RADIUS = 5  # Reduced to 5 for speed based on sensitivity analysis
NODATA_VAL = -9999.0

REGIONS_INFO = {
    "Japan": {'nc': 'japan.nc', 'tif': 'japan_mtbmap.tif'},
    "New_York": {'nc': 'new_york.nc', 'tif': 'new_york_mtbmap.tif'},
    "Melbourne": {'nc': 'melbourne.nc', 'tif': 'melbourne_mtbmap.tif'},
    "Guangzhou_Bay_Area": {'nc': 'guangzhou_bay.nc', 'tif': 'guangzhou_bay_mtbmap.tif'},
    "Hangzhou_Bay": {'nc': 'hangzhou_bay.nc', 'tif': 'hangzhou_bay_mtbmap.tif'}
}

# --- 3. 辅助函数 ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return (img > 128).astype(np.uint8)

def save_geotiff(array, ref_path, out_path, nodata_val):
    try:
        with rasterio.open(ref_path) as ref: 
            profile = ref.profile
        profile.update(dtype='float32', count=1, nodata=nodata_val)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(array.astype(np.float32), 1)
        # print(f"   GeoTIFF saved: {out_path}")
    except Exception as e: print(f"Error saving Tiff: {e}")

def save_visual_png(data, out_path, title, cmap='viridis'):
    # --- 绘图配置升级 (大字号) ---
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    height, width = data.shape
    aspect_ratio = height / width
    if aspect_ratio > 1.5: figsize = (8, 10); shrink = 0.5
    elif aspect_ratio < 0.67: figsize = (12, 6); shrink = 0.8
    else: figsize = (10, 8); shrink = 0.7
    
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

def interpolate_dataarray_to_raster(data_array, target_shape):
    source_lat = data_array['latitude'].values
    source_lon = data_array['longitude'].values
    source_vals = data_array.values
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
    points = np.vstack((source_lat_grid.ravel(), source_lon_grid.ravel())).T
    values = source_vals.ravel()
    valid_mask = ~np.isnan(values)
    
    if not np.any(valid_mask): return np.zeros(target_shape)

    # Note: Assuming simple linear lat/lon space for target (approx)
    # Getting bounds from lat/lon min/max is risky if projection differs significantly
    # BUT keeping logic untouched as requested.
    target_lat_vec = np.linspace(source_lat.max(), source_lat.min(), target_shape[0])
    target_lon_vec = np.linspace(source_lon.min(), source_lon.max(), target_shape[1])
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon_vec, target_lat_vec)
    
    interp_values = griddata(points[valid_mask], values[valid_mask], (target_lat_grid, target_lon_grid), method='nearest')
    return interp_values

# --- 4. 核心处理逻辑 ---

def process_region(region_name):
    print(f"\n--- Model B Processing: {region_name} ---")
    
    # Paths
    file_info = REGIONS_INFO[region_name]
    nc_path = os.path.join(config.NC_DIR, file_info['nc'])
    ref_tiff_path = os.path.join(config.TIFF_DIR, file_info['tif'])
    
    geometry_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
    output_dir = os.path.join(config.OUTPUT_ROOT, 'model_b_results', region_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    try:
        # 1. Load Geometry
        land_mask = load_image(os.path.join(geometry_dir, 'binary_land_mask.png'))
        coast_mask = load_image(os.path.join(geometry_dir, 'raster_coastline.png'))
        
        # 2. Load Wave Data & Interpolate
        print("   Preparing dense H_mean field...")
        wave_ds = xr.open_dataset(nc_path)
        h_mean_low_res = wave_ds[WAVE_VARIABLE].mean(dim='time', skipna=True)
        h_mean_dense = interpolate_dataarray_to_raster(h_mean_low_res, land_mask.shape)
        h_mean_dense[land_mask == 1] = 0 # Ensure land is 0 for water lookups
        wave_ds.close()

        # 3. Neighborhood Analysis Loop
        print(f"   Starting Neighborhood Analysis (R={NEIGHBORHOOD_RADIUS})...")
        _, indices = distance_transform_edt(~coast_mask.astype(bool), return_indices=True)
        
        inland_influence_map = np.zeros_like(land_mask, dtype=np.float32)
        land_pixels_r, land_pixels_c = np.where(land_mask == 1)

        total_pixels = len(land_pixels_r)
        start_time = time.time()
        
        # Optimization: Don't print too often
        print_interval = max(1, total_pixels // 5) 

        for i in range(total_pixels):
            if (i + 1) % print_interval == 0:
                print(f"     Processed {i+1}/{total_pixels} pixels...")

            r, c = land_pixels_r[i], land_pixels_c[i]
            nearest_coast_r, nearest_coast_c = indices[0, r, c], indices[1, r, c]
            
            r_min = max(0, nearest_coast_r - NEIGHBORHOOD_RADIUS)
            r_max = min(land_mask.shape[0], nearest_coast_r + NEIGHBORHOOD_RADIUS + 1)
            c_min = max(0, nearest_coast_c - NEIGHBORHOOD_RADIUS)
            c_max = min(land_mask.shape[1], nearest_coast_c + NEIGHBORHOOD_RADIUS + 1)
            
            window = h_mean_dense[r_min:r_max, c_min:c_max]
            valid_pixels = window[window > 0]
            
            if valid_pixels.size > 0:
                inland_influence_map[r, c] = np.mean(valid_pixels)
        
        end_time = time.time()
        print(f"   Loop finished in {end_time - start_time:.1f}s.")
        
        inland_influence_map[land_mask == 0] = NODATA_VAL

        # 4. Save
        title = f"Model B: Neighborhood (R={NEIGHBORHOOD_RADIUS}) - {region_name}"
        png_path = os.path.join(output_dir, f"inland_influence_neighborhood_r{NEIGHBORHOOD_RADIUS}.png")
        tif_path = os.path.join(output_dir, f"inland_influence_neighborhood_r{NEIGHBORHOOD_RADIUS}.tif")
        
        save_visual_png(inland_influence_map, png_path, title)
        save_geotiff(inland_influence_map, ref_tiff_path, tif_path, NODATA_VAL)
        
        print(f"✅ Model B finished for {region_name}")

    except Exception as e:
        print(f"❌ Error in Model B for {region_name}: {e}")
        import traceback
        traceback.print_exc()

# --- 5. 主程序 ---
def main():
    print(f"\n🚀 Step 6: Running Model B (Neighborhood Analysis)...")
    for region in REGIONS_INFO.keys():
        process_region(region)
    print(f"\n✨ Step 6 Completed.")

if __name__ == '__main__':
    main()