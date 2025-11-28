# filename: 7_model_fusion.py
# description: (Refactored) Final Model Fusion.
#              Combines Model A (Propagation) and Model B (Neighborhood) results.

import numpy as np
import os
import sys
import rasterio
import matplotlib.pyplot as plt

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 参数配置 ---
WEIGHT_PROPAGATION = 0.5 
WEIGHT_NEIGHBORHOOD = 0.5 
NODATA_VAL = -9999.0

# 重要：这里必须和 Model B 中使用的半径一致
NEIGHBORHOOD_RADIUS = 5 

REGIONS_TO_PROCESS = ['Japan', 'Hangzhou_Bay', 'Guangzhou_Bay_Area', 'Melbourne', 'New_York']

# --- 3. 辅助函数 ---

def create_output_directory(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def save_geotiff(array, ref_path, out_path, nodata_val):
    try:
        with rasterio.open(ref_path) as ref:
            profile = ref.profile
        profile.update(dtype='float32', count=1, nodata=nodata_val)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(array.astype(np.float32), 1)
        # print(f"   GeoTIFF saved: {out_path}")
    except Exception as e: print(f"Error saving GeoTIFF: {e}")

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

# --- 4. 核心处理逻辑 ---

def process_region(region_name):
    print(f"\n--- Fusing Models for: {region_name} ---")
    
    try:
        # Paths
        # Model A Input
        model_a_dir = os.path.join(config.OUTPUT_ROOT, 'model_a_results', region_name)
        prop_file = os.path.join(model_a_dir, 'inland_influence_mean_only.tif')
        
        # Model B Input
        model_b_dir = os.path.join(config.OUTPUT_ROOT, 'model_b_results', region_name)
        neigh_file = os.path.join(model_b_dir, f'inland_influence_neighborhood_r{NEIGHBORHOOD_RADIUS}.tif')
        
        # Ref Tiff
        # Handling naming inconsistencies (sorry, hardcoding for safety)
        ref_name = f"{region_name.lower()}_mtbmap.tif"
        if region_name == 'New_York': ref_name = 'new_york_mtbmap.tif'
        elif region_name == 'Guangzhou_Bay_Area': ref_name = 'guangzhou_bay_mtbmap.tif'
        elif region_name == 'Hangzhou_Bay': ref_name = 'hangzhou_bay_mtbmap.tif'
        ref_tiff_path = os.path.join(config.TIFF_DIR, ref_name)
        
        # Output
        output_dir = os.path.join(config.OUTPUT_ROOT, 'final_combined_outputs', region_name)
        create_output_directory(output_dir)

        # Check Inputs
        if not os.path.exists(prop_file):
            print(f"⚠️ Missing Model A result: {prop_file}"); return
        if not os.path.exists(neigh_file):
            print(f"⚠️ Missing Model B result: {neigh_file}"); return

        # Load Data
        with rasterio.open(prop_file) as src:
            prop_map = src.read(1)
            prop_nodata = src.nodata or NODATA_VAL
            
        with rasterio.open(neigh_file) as src:
            neigh_map = src.read(1)
            neigh_nodata = src.nodata or NODATA_VAL
        
        # Core Logic
        # Create Master Mask from Model A (Propagation)
        master_mask = (prop_map != prop_nodata) & (prop_map > 0)
        
        # Clean Data (replace nodata with 0 for weighted average)
        prop_safe = np.where(master_mask, prop_map, 0)
        neigh_safe = np.where((neigh_map != neigh_nodata) & (neigh_map > 0), neigh_map, 0)
        
        # Weighted Average
        combined_map = (WEIGHT_PROPAGATION * prop_safe) + (WEIGHT_NEIGHBORHOOD * neigh_safe)
        
        # Apply Mask
        final_map = np.where(master_mask, combined_map, NODATA_VAL)
        
        # Save
        suffix = f"w{int(WEIGHT_PROPAGATION*10)}_{int(WEIGHT_NEIGHBORHOOD*10)}"
        title = f"Final Combined Influence - {region_name}"
        
        save_visual_png(final_map, os.path.join(output_dir, f"final_influence_{suffix}.png"), title)
        save_geotiff(final_map, ref_tiff_path, os.path.join(output_dir, f"final_influence_{suffix}.tif"), NODATA_VAL)
        
        print(f"✅ Fusion complete for {region_name}")

    except Exception as e:
        print(f"❌ Error in fusion: {e}")
        import traceback
        traceback.print_exc()

# --- 5. 主程序 ---
def main():
    print(f"\n🚀 Step 7: Final Model Fusion...")
    for region in REGIONS_TO_PROCESS:
        process_region(region)
    print(f"\n✨ Step 7 Completed.")

if __name__ == '__main__':
    main()