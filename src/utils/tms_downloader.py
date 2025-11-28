# filename: tms_downloader.py
# description: Utility to download TMS map tiles for the study regions.
#              Saves tiles to 'data/tms_tiles/' for use in the mosaic step.

import os
import sys
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 配置 ---

# Tile URL templates
LOBELIA_URL_TEMPLATE = "https://tiles.lobelia.earth/lobelia-oceans-dark-v2/3857/{z}/{x}/{y}.png"
MTBMAP_URL_TEMPLATE = "https://mtbmap.org/osm_tiles2/{z}/{x}/{y}.png"

# Regions with their x and y ranges at zoom level 7
# (Same definitions as used in mosaic script)
REGIONS = {
    "japan": {"x_range": (112, 113), "y_range": (50, 50)},
    "hangzhou_bay": {"x_range": (106, 107), "y_range": (52, 52)},
    "guangzhou_bay": {"x_range": (104, 104), "y_range": (55, 56)},
    "melbourne": {"x_range": (115, 115), "y_range": (78, 78)},
    "new_york": {"x_range": (37, 38), "y_range": (48, 48)}
}

# Zoom level
ZOOM = 7

# --- 3. 核心逻辑 ---

def download_tile(url_template, x, y, tile_type):
    """Download a single tile and save it to the appropriate subdirectory."""
    url = url_template.format(z=ZOOM, x=x, y=y)
    
    # 动态构建路径: data/tms_tiles/lobelia/7_x_y.png
    output_dir = os.path.join(config.TMS_DIR, tile_type)
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{ZOOM}_{x}_{y}.png")
    
    # Skip if file already exists
    if os.path.exists(file_path):
        # print(f"Skipping existing tile: {file_path}")
        return
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"   Downloaded: {file_path}")
        else:
            print(f"❌ Failed: {url} (Status: {response.status_code})")
            
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.2)
        
    except Exception as e:
        print(f"❌ Error downloading {url}: {str(e)}")

def main():
    print(f"\n🚀 Starting TMS Tile Download...")
    print(f"   Target Directory: {config.TMS_DIR}")
    
    # Prepare all download tasks
    tasks = []
    
    # Create tasks for each region for both tile sources
    for region_name, coordinates in REGIONS.items():
        x_start, x_end = coordinates["x_range"]
        y_start, y_end = coordinates["y_range"]
        
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                # Add task for Lobelia
                tasks.append((LOBELIA_URL_TEMPLATE, x, y, "lobelia"))
                # Add task for MTB Map (same coordinates)
                tasks.append((MTBMAP_URL_TEMPLATE, x, y, "mtbmap"))
    
    # Print summary
    print(f"   Total tiles to check/download: {len(tasks)}")
    
    # Use ThreadPoolExecutor to download tiles concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        for task in tasks:
            executor.submit(download_tile, *task)
    
    print("\n✨ Download completed!")

if __name__ == "__main__":
    main()