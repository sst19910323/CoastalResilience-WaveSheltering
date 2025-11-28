# filename: calculate_effective_wave_component.py
# description: [EXPERIMENTAL] Attempt to calculate "Effective Wave Height" perpendicular to coastline.
#              This approach was explored but not included in the final paper due to 
#              sensitivity to low-resolution global wave direction data.

import os
import sys
import time
import netCDF4 as nc
import numpy as np
import rasterio
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
import traceback
import matplotlib.pyplot as plt

# --- 1. 引入路径配置 ---
# 实验文件夹在根目录下，所以 sys.path 需要添加上一级目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 假设 src 下有 config，我们需要把 src 加入 path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config

# --- 2. 参数配置 ---
# 仅在部分区域测试过
REGIONS_TO_PROCESS = ['Melbourne', 'New_York']

# 路径映射 (利用 config)
REGION_GEOMETRY_FOLDER_MAP = {
    'Japan': 'Japan', 
    'Hangzhou_Bay': 'Hangzhou_Bay', 
    'Guangzhou_Bay_Area': 'Guangzhou_Bay_Area',
    'Melbourne': 'Melbourne', 
    'New_York': 'New_York'
}

OUTPUT_FILENAME_TIF = 'VHM0_effective_mean_experimental.tif'
OUTPUT_FILENAME_PNG = 'VHM0_effective_mean_experimental.png'
OUTPUT_CMAP = 'inferno'
NODATA_VAL = -9999.0
NPY_NORMAL_DIR_NODATA = -9999.0
NPY_BUFFER_NODATA = 0

# --- 3. 辅助函数 ---

def create_output_directory(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def save_array_as_png(data_array, nodata_val, output_path, title, cmap='viridis'):
    plt.figure(figsize=(10, 8)); ax = plt.gca()
    plot_data = data_array.copy().astype(float)
    plot_data[plot_data <= 0] = np.nan 
    
    if np.all(np.isnan(plot_data)):
        ax.text(0.5, 0.5, 'All data is NoData', ha='center', va='center', transform=ax.transAxes)
    else:
        current_cmap = plt.get_cmap(cmap).copy(); current_cmap.set_bad(color='lightgray')
        im = ax.imshow(plot_data, cmap=current_cmap)
        plt.colorbar(im, label=title.split(' - ')[0])
    
    ax.set_title(title + " (Experimental)"); ax.set_axis_off()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1); plt.close()
    print(f"   Visual PNG saved: {os.path.basename(output_path)}")

def save_geotiff(array, ref_profile, out_path, nodata_val):
    profile = ref_profile.copy(); profile.update(dtype='float32', count=1, nodata=nodata_val)
    with rasterio.open(out_path, 'w', **profile) as dst: dst.write(array.astype(np.float32), 1)

def fill_gaps_nearest(array_with_gaps, nodata_value, land_mask):
    # 简单的填补逻辑
    gaps_in_water_mask = ((array_with_gaps == nodata_value) | np.isnan(array_with_gaps)) & ~land_mask
    if not np.any(gaps_in_water_mask): return array_with_gaps

    valid_mask = (array_with_gaps != nodata_value) & ~np.isnan(array_with_gaps)
    if not np.any(valid_mask): return array_with_gaps
        
    _, indices = distance_transform_edt(valid_mask, return_indices=True)
    
    filled_array = array_with_gaps.copy()
    gap_r, gap_c = np.where(gaps_in_water_mask)
    source_r, source_c = indices[0, gap_r, gap_c], indices[1, gap_r, gap_c]
    
    filled_array[gap_r, gap_c] = filled_array[source_r, source_c]
    return filled_array

# --- 4. 核心逻辑 ---

def process_region(region_name):
    print(f"\n--- Processing Experimental Region: {region_name} ---")
    
    try:
        # Paths
        nc_file = os.path.join(config.NC_DIR, f"{config.REGIONS_CONFIG[region_name]['nc_file']}") # 需要确保 config 里有这个映射
        # 如果 config 里没有反向映射，这里简写一下：
        if region_name == 'New_York': nc_file = os.path.join(config.NC_DIR, 'new_york.nc')
        elif region_name == 'Melbourne': nc_file = os.path.join(config.NC_DIR, 'melbourne.nc')
        
        # Ref Tiff
        ref_tiff = os.path.join(config.TIFF_DIR, f"{region_name.lower()}_mtbmap.tif")
        if region_name == 'New_York': ref_tiff = os.path.join(config.TIFF_DIR, 'new_york_mtbmap.tif') # 你的文件名也是乱的，如果不统一需要手动搞
        
        # Geometry Inputs (From Step 2)
        geometry_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
        coastal_normal_npy = os.path.join(geometry_dir, "propagated_offshore_direction_raw.npy")
        buffer_npy = os.path.join(geometry_dir, "unified_buffer_level_map_raw.npy")
        
        # Output
        exp_output_dir = os.path.join(config.OUTPUT_ROOT, 'experimental_outputs', region_name)
        create_output_directory(exp_output_dir)

        # Load Data
        with rasterio.open(ref_tiff) as ref_ds:
            target_profile = ref_ds.profile
            target_height, target_width = ref_ds.height, ref_ds.width
            land_mask = (ref_ds.read(1) == 1)
            
        coastal_normal_map = np.load(coastal_normal_npy)
        buffer_map = np.load(buffer_npy)
        
        with nc.Dataset(nc_file, 'r') as ds:
            vhm0_ts = ds.variables['VHM0']
            vmdr_ts = ds.variables['VMDR']
            num_timesteps = vhm0_ts.shape[0]
            
            # Grid Setup
            lats = ds.variables['latitude'][:]
            lons = ds.variables['longitude'][:]
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            src_points = np.vstack((lat_grid.ravel(), lon_grid.ravel())).T
            
            tgt_lat = np.linspace(lats.max(), lats.min(), target_height)
            tgt_lon = np.linspace(lons.min(), lons.max(), target_width)
            tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)
            
            total_effective = np.zeros((target_height, target_width))
            
            # Loop
            print(f"   Looping through {num_timesteps} time steps (this takes time)...")
            for t in range(num_timesteps):
                if t % 50 == 0: print(f"   Step {t}/{num_timesteps}")
                
                # Interpolate Frame
                h_t = griddata(src_points, vhm0_ts[t].ravel(), (tgt_lat_grid, tgt_lon_grid), method='nearest')
                dir_t = griddata(src_points, vmdr_ts[t].ravel(), (tgt_lat_grid, tgt_lon_grid), method='nearest')
                
                # Calc Effective Component
                eff_t = np.zeros_like(h_t)
                mask = (buffer_map > 0) & ~np.isnan(h_t) & ~np.isnan(dir_t) & (coastal_normal_map != -9999) & ~land_mask
                
                if np.any(mask):
                    h_val = h_t[mask]
                    wave_meteo = dir_t[mask]
                    normal = coastal_normal_map[mask]
                    
                    wave_math = (270 - wave_meteo + 360) % 360
                    delta = wave_math - normal
                    cos_factor = np.maximum(0, np.cos(np.deg2rad(delta)))
                    
                    eff_t[mask] = h_val * cos_factor
                
                total_effective += eff_t
            
            # Finalize
            mean_eff = total_effective / num_timesteps
            mean_eff[land_mask] = NODATA_VAL
            mean_eff_filled = fill_gaps_nearest(mean_eff, NODATA_VAL, land_mask)
            
            save_geotiff(mean_eff_filled, target_profile, os.path.join(exp_output_dir, OUTPUT_FILENAME_TIF), NODATA_VAL)
            save_array_as_png(mean_eff_filled, NODATA_VAL, os.path.join(exp_output_dir, OUTPUT_FILENAME_PNG), 
                            f"Effective Wave (Exp) - {region_name}", cmap=OUTPUT_CMAP)
            
    except Exception as e:
        print(f"❌ Error in experimental script for {region_name}: {e}")

if __name__ == '__main__':
    for r in REGIONS_TO_PROCESS:
        process_region(r)