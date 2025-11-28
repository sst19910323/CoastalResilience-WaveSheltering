# filename: 4_process_wave_data.py
# description: (Refactored) Calculates wave statistics (Mean, P95) from NC files
#              and reprojects them to align perfectly with the reference GeoTIFFs.

import os
import sys

# --- 1. PROJ_LIB FIX (KEEP IT) ---
try:
    import pyproj
    pyproj_datadir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = os.path.join(pyproj_datadir)
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not set PROJ_LIB: {e}")

import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine
import traceback
import matplotlib.pyplot as plt

# --- 2. 引入配置 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# --- 3. 参数配置 ---
# 地区映射：脚本用的名字 -> 文件夹/文件名
REGIONS_TO_PROCESS = ['Japan', 'Hangzhou_Bay', 'Guangzhou_Bay_Area', 'Melbourne', 'New_York']

# 输入文件名映射 (Region -> NC Filename, Tiff Filename)
REGION_FILE_MAP = {
    'Japan': {'nc': 'japan.nc', 'tif': 'japan_mtbmap.tif'},
    'Hangzhou_Bay': {'nc': 'hangzhou_bay.nc', 'tif': 'hangzhou_bay_mtbmap.tif'},
    'Guangzhou_Bay_Area': {'nc': 'guangzhou_bay.nc', 'tif': 'guangzhou_bay_mtbmap.tif'},
    'Melbourne': {'nc': 'melbourne.nc', 'tif': 'melbourne_mtbmap.tif'},
    'New_York': {'nc': 'new_york.nc', 'tif': 'new_york_mtbmap.tif'}
}

VAR_CONFIG = {
    'VHM0_MEAN': {
        'output_aligned_filename': 'VHM0_aligned.tif',
        'png_aligned_filename': 'VHM0_aligned.png',
        'nodata_val': -9999.0,
        'cmap': 'viridis'
    },
    'VHM0_P95': {
        'output_aligned_filename': 'VHM0_P95_aligned.tif',
        'png_aligned_filename': 'VHM0_P95_aligned.png',
        'nodata_val': -9999.0,
        'cmap': 'viridis'
    }
}

RESAMPLING_METHOD = Resampling.bilinear
NC_SRC_CRS = CRS.from_epsg(4326)
TARGET_CRS_EXPECTED = CRS.from_epsg(3857)

# --- 4. 辅助函数 ---

def create_output_directory(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def save_array_as_png(data_array, nodata_val, output_path, title, cmap='viridis'):
    # Masking
    masked_array = np.ma.masked_where((data_array == nodata_val) | np.isnan(data_array), data_array)
    
    if masked_array.count() == 0:
        print(f"   Skipping PNG for {title}, all data masked.")
        return

    # --- 绘图配置升级 (大字号) ---
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16

    h, w = data_array.shape
    aspect = h / w if w != 0 else 1.0
    plt.figure(figsize=(10, 10 * aspect), dpi=150)
    
    plt.imshow(masked_array, cmap=cmap)
    plt.colorbar(label=title, pad=0.02)
    plt.title(title)
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Visual PNG saved: {os.path.basename(output_path)}")

# --- 5. 核心处理逻辑 ---

def process_region(region_name):
    print(f"\n--- Processing Region: {region_name} ---")

    # 路径构造
    file_info = REGION_FILE_MAP[region_name]
    nc_file_path = os.path.join(config.NC_DIR, file_info['nc'])
    reference_tiff_path = os.path.join(config.TIFF_DIR, file_info['tif'])
    
    # 输出到 'wave_data_processed/[Region]' -> 改名为 aligned_output 比较好懂
    # 但为了保持兼容，我们还是用 wave_data_processed 放在 outputs 下
    region_output_dir = os.path.join(config.OUTPUT_ROOT, 'wave_data_processed', region_name)
    create_output_directory(region_output_dir)

    # Check Inputs
    if not os.path.exists(nc_file_path):
        print(f"❌ Missing NC file: {nc_file_path}"); return
    if not os.path.exists(reference_tiff_path):
        print(f"❌ Missing Tiff file: {reference_tiff_path}"); return

    # 1. Read Ref Tiff Info
    try:
        with rasterio.open(reference_tiff_path) as ref_ds:
            target_transform = ref_ds.transform
            target_width = ref_ds.width
            target_height = ref_ds.height
            target_crs_to_use = ref_ds.crs or TARGET_CRS_EXPECTED
            ref_data = ref_ds.read(1)
            land_mask = (ref_data == 1) # Assumes 1 is land?
            print(f"   Reference loaded: {file_info['tif']}")
    except Exception as e:
        print(f"❌ Error reading TIFF: {e}"); return

    # 2. Process NetCDF
    try:
        with nc.Dataset(nc_file_path, 'r') as dataset:
            if 'VHM0' not in dataset.variables:
                print(f"❌ VHM0 not found in {file_info['nc']}"); return

            print(f"   Calculating stats from NC...")
            vhm0_data = dataset.variables['VHM0'][:, :, :]
            
            # Mean
            masked_vhm0 = np.ma.masked_invalid(vhm0_data)
            vhm0_mean = np.ma.mean(masked_vhm0, axis=0).filled(np.nan)
            
            # P95 (Memory Safe Loop)
            vhm0_p95 = np.full(vhm0_mean.shape, np.nan, dtype=np.float32)
            time_steps = vhm0_data.shape[0]
            # 优化：如果是 masked array，可以直接用 np.percentile 但要注意 invalid
            # 这里的循环虽然慢但是稳，原样保留
            for i in range(vhm0_data.shape[1]):
                for j in range(vhm0_data.shape[2]):
                    pixel_ts = masked_vhm0[:, i, j]
                    if pixel_ts.count() > 0:
                        vhm0_p95[i, j] = np.percentile(pixel_ts.compressed(), 95)
            
            stats = {'VHM0_MEAN': vhm0_mean, 'VHM0_P95': vhm0_p95}

            # Get Source Grid Info
            lats = dataset.variables['latitude'][:]
            lons = dataset.variables['longitude'][:]
            lat_desc = lats[0] > lats[-1]
            lon_res = abs(lons[1] - lons[0])
            lat_res = abs(lats[1] - lats[0])
            
            # Build Source Transform (Affine)
            # transform = translation(min_lon, max_lat) * scale(res, -res)
            src_transform = Affine.translation(lons.min() - lon_res/2, lats.max() + lat_res/2) * Affine.scale(lon_res, -lat_res)

            # 3. Reproject Loop
            for key, data in stats.items():
                cfg = VAR_CONFIG[key]
                nodata = cfg['nodata_val']
                
                # Flip if needed
                src_data = data.copy()
                if not lat_desc:
                    src_data = np.ascontiguousarray(np.flipud(src_data))
                
                dest_array = np.full((target_height, target_width), nodata, dtype=np.float32)
                
                reproject(
                    source=src_data,
                    destination=dest_array,
                    src_transform=src_transform,
                    src_crs=NC_SRC_CRS,
                    dst_transform=target_transform,
                    dst_crs=target_crs_to_use,
                    resampling=RESAMPLING_METHOD
                )
                
                # Mask Land
                # dest_array[land_mask] = nodata # Optional: Mask land?
                # 原逻辑似乎是 mask land，这里保留
                dest_array[land_mask] = nodata

                # Save Tif
                out_tif = os.path.join(region_output_dir, cfg['output_aligned_filename'])
                with rasterio.open(out_tif, 'w', driver='GTiff', 
                                 height=target_height, width=target_width, 
                                 count=1, dtype=str(dest_array.dtype), 
                                 crs=target_crs_to_use, transform=target_transform, 
                                 nodata=nodata) as dst:
                    dst.write(dest_array, 1)
                
                # Save Png
                if cfg.get('png_aligned_filename'):
                    out_png = os.path.join(region_output_dir, cfg['png_aligned_filename'])
                    save_array_as_png(dest_array, nodata, out_png, f"{key} Aligned", cfg['cmap'])
                    
    except Exception as e:
        print(f"❌ Error processing {region_name}: {e}"); traceback.print_exc()

# --- 6. 主程序 ---
def main():
    print(f"\n🚀 Step 4: Processing and Aligning Wave Data...")
    for region in REGIONS_TO_PROCESS:
        process_region(region)
    print(f"\n✨ Step 4 Completed.")

if __name__ == '__main__':
    main()