# filename: analyze_poi.py
# description: (Refactored) Point of Interest (POI) Analysis.
#              Generates Time-Series, Boxplots, and Wave Roses for specific study locations.
#              Produces Figure 6 & 7 for the manuscript with publication-quality formatting.

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- 2. 绘图配置升级 (PUBLICATION QUALITY) ---
# 暴力加大字号，解决 "visual aspect" 投诉
PAPER_FONT_SIZE = 16
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': PAPER_FONT_SIZE,
    'axes.labelsize': PAPER_FONT_SIZE + 2,
    'axes.titlesize': PAPER_FONT_SIZE + 4,
    'xtick.labelsize': PAPER_FONT_SIZE,
    'ytick.labelsize': PAPER_FONT_SIZE,
    'legend.fontsize': PAPER_FONT_SIZE,
    'figure.titlesize': PAPER_FONT_SIZE + 6,
    'lines.linewidth': 2.5,       # 加粗线条
    'axes.linewidth': 1.5,        # 加粗坐标轴
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'grid.linewidth': 1.0,
    'savefig.dpi': 300            # 高清输出
})

try:
    from windrose import WindroseAxes
    WINDROSE_AVAILABLE = True
except ImportError:
    WINDROSE_AVAILABLE = False
    print("⚠️ Warning: 'windrose' library not installed. Wave roses will be skipped.")

# --- 3. 区域与站点配置 (保留原有的经纬度) ---
REGIONS_CONFIG = {
    "Japan": {
        "nc_file": "japan.nc",
        "locations": {
            "Tokyo Bay (Sheltered)": {"lat": 35.5804, "lon": 139.8759, "color": "blue"},
            "Offshore Chiba (Open)": {"lat": 35.0529, "lon": 140.1661, "color": "red"},
            "Ise Bay (Sheltered)": {"lat": 34.9570, "lon": 136.7434, "color": "green"},
            "South of Izu (Open)": {"lat": 34.5540, "lon": 138.8474, "color": "purple"}
        }
    },
    "New_York": {
        "nc_file": "new_york.nc",
        "locations": {
            "New York Harbor (Sheltered)": {"lat": 40.50498, "lon": -74.10114, "color": "blue"},
            "Baltimore Approach (Sheltered)": {"lat": 40.55208, "lon": -74.04636, "color": "green"},
            "Long Island South (Open)": {"lat": 40.72947, "lon": -72.71663, "color": "red"},
            "Assateague Island (Open)": {"lat": 38.06024, "lon": -75.08698, "color": "purple"}
        }
    },
    "Melbourne": {
        "nc_file": "melbourne.nc",
        "locations": {
            "Port Phillip Bay (Sheltered)": {"lat": -37.98841, "lon": 144.86771, "color": "blue"},
            "East of Otway NP (Open)": {"lat": -38.75124, "lon": 143.92088, "color": "red"}
        }
    },
    "Guangzhou_Bay_Area": {
        "nc_file": "guangzhou_bay.nc",
        "locations": {
            "Guangzhou Bay (Sheltered)": {"lat": 22.65259, "lon": 113.72617, "color": "blue"},
            "Shenzhen Bay (Sheltered)": {"lat": 22.43495, "lon": 113.90467, "color": "green"},
            "Hong Kong North (Sheltered)": {"lat": 22.30442, "lon": 114.11364, "color": "navy"},
            "Hong Kong South (Open)": {"lat": 22.15711, "lon": 114.20367, "color": "red"},
            "Shanwei Coast (Open)": {"lat": 22.76942, "lon": 115.25543, "color": "purple"}
        }
    },
    "Hangzhou_Bay": {
        "nc_file": "hangzhou_bay.nc",
        "locations": {
            "South of Shanghai (Sheltered)": {"lat": 30.76240, "lon": 121.69209, "color": "blue"},
            "North of Ningbo (Sheltered)": {"lat": 30.15454, "lon": 121.68893, "color": "green"},
            "Xiangshan Harbor (Sheltered)": {"lat": 29.82179, "lon": 122.10049, "color": "navy"},
            "South of Zhoushan (Open)": {"lat": 29.89223, "lon": 122.19153, "color": "red"},
            "East of Zhoushan (Open)": {"lat": 30.05271, "lon": 122.59414, "color": "purple"}
        }
    }
}

# 坐标变量名
LAT_COORD = 'latitude'
LON_COORD = 'longitude'

# --- 4. 核心逻辑 ---

def load_and_extract_data(file_path, locations):
    print(f"   Loading dataset: {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return None, None
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None

    all_point_data = {}
    for loc_name, cfg in locations.items():
        try:
            # Nearest neighbor selection
            point_ds = ds.sel({LAT_COORD: cfg['lat'], LON_COORD: cfg['lon']}, method='nearest')
            all_point_data[loc_name] = point_ds
        except Exception as e:
            print(f"   ⚠️ Failed to extract for {loc_name}: {e}")
            all_point_data[loc_name] = None
    
    return ds, all_point_data

def calculate_summary_statistics(all_point_data):
    stats_list = []
    STATS_PARAMS = ['VHM0', 'VTPK', 'VMXL'] 
    
    for loc_name, point_ds in all_point_data.items():
        if point_ds is None: continue
        summary = {"Location": loc_name}
        for param in STATS_PARAMS:
            if param in point_ds:
                data = point_ds[param].to_pandas().dropna()
                if not data.empty:
                    summary[f'{param}_Mean'] = data.mean()
                    summary[f'{param}_P95'] = data.quantile(0.95)
                    summary[f'{param}_Max'] = data.max()
        stats_list.append(summary)
            
    stats_df = pd.DataFrame(stats_list)
    if not stats_df.empty:
        stats_df = stats_df.set_index("Location")
    return stats_df

# --- 5. 绘图函数 (Enhanced) ---

def plot_comparative_timeseries(all_point_data, param, locations_config, output_dir, ds_ref):
    """绘制时间序列对比图 (Fig 6 style)"""
    fig, ax = plt.subplots(figsize=(14, 8)) # 宽一点
    
    has_data = False
    for loc_name, point_ds in all_point_data.items():
        if point_ds is not None and param in point_ds:
            series = point_ds[param]
            # Use pandas plotting for better date handling
            series.to_pandas().plot(ax=ax, label=loc_name, color=locations_config[loc_name]['color'], alpha=0.8, linewidth=2)
            has_data = True
    
    if not has_data: return

    param_units = ds_ref[param].attrs.get('units', '')
    ax.set_title(f"Time Series: {param}", pad=20)
    ax.set_ylabel(f"{param} ({param_units})")
    ax.set_xlabel("Date")
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 强制裁切白边
    plt.savefig(os.path.join(output_dir, f"Fig_Timeseries_{param}.png"), bbox_inches='tight')
    plt.close()

def plot_comparative_boxplots(all_point_data, param, locations_config, output_dir, ds_ref):
    """绘制箱线图 (Fig 7 style)"""
    data_to_plot, labels, colors = [], [], []
    
    # 按照 Sheltered 和 Open 分组排序 (可选)
    sorted_locs = sorted(all_point_data.keys(), key=lambda x: "Open" in x) # Put Sheltered first usually
    
    for loc_name in sorted_locs:
        point_ds = all_point_data.get(loc_name)
        if point_ds is not None and param in point_ds:
            series = point_ds[param].to_pandas().dropna()
            if not series.empty:
                data_to_plot.append(series)
                # 简化标签：去掉括号里的内容，太长了，图例已经解释了颜色
                short_name = loc_name.split('(')[0].strip()
                labels.append(short_name)
                colors.append(locations_config[loc_name]['color'])
                
    if not data_to_plot: return

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 画箱线图
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    medianprops={'color':'black', 'linewidth': 2},
                    boxprops={'linewidth': 1.5},
                    whiskerprops={'linewidth': 1.5},
                    capprops={'linewidth': 1.5})
                    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    param_units = ds_ref[param].attrs.get('units', '')
    ax.set_title(f"Distribution: {param}", pad=20)
    ax.set_ylabel(f"{param} ({param_units})")
    
    # 旋转标签防止重叠
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f"Fig_Boxplot_{param}.png"), bbox_inches='tight')
    plt.close()

def plot_wave_rose(point_ds, loc_name, output_dir, ds_ref):
    """绘制玫瑰图"""
    if not WINDROSE_AVAILABLE or point_ds is None: return
    
    strength_param, dir_param = 'VHM0', 'VMDR'
    if strength_param not in point_ds or dir_param not in point_ds: return

    wd = point_ds[dir_param].to_pandas().dropna()
    ws = point_ds[strength_param].to_pandas().loc[wd.index].dropna()

    if wd.empty or ws.empty: return

    # Windrose 比较特殊，需要特殊的字号设置
    fig = plt.figure(figsize=(10, 10))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect)
    fig.add_axes(ax)
    
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', nsector=16)
    
    strength_units = ds_ref[strength_param].attrs.get('units', '')
    # Legend
    ax.set_legend(title=f'{strength_param} ({strength_units})', 
                  loc='best', fontsize=PAPER_FONT_SIZE-2)
    
    plt.title(f"{loc_name}", fontsize=PAPER_FONT_SIZE+4, y=1.08)
    
    safe_loc_name = "".join(c for c in loc_name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f"Fig_WaveRose_{safe_loc_name}.png"), bbox_inches='tight')
    plt.close()

# --- 6. 主流程 ---

def process_region(region_name, cfg):
    print(f"\n--- Analyzing POIs for: {region_name} ---")
    
    nc_path = os.path.join(config.NC_DIR, cfg['nc_file'])
    locations = cfg['locations']
    
    # 输出到专门的 plot 文件夹
    output_dir = os.path.join(config.OUTPUT_ROOT, 'wave_analysis_plots', region_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. Load
    master_ds, all_point_data = load_and_extract_data(nc_path, locations)
    if master_ds is None: return

    # 2. Statistics CSV
    stats_df = calculate_summary_statistics(all_point_data)
    stats_csv_path = os.path.join(output_dir, "Table_Summary_Statistics.csv")
    stats_df.to_csv(stats_csv_path)
    print(f"   Saved statistics to {os.path.basename(stats_csv_path)}")
    
    # 3. Plots
    PARAMS_TO_PLOT = ['VHM0', 'VTPK']
    
    for param in PARAMS_TO_PLOT:
        if param in master_ds:
            plot_comparative_timeseries(all_point_data, param, locations, output_dir, master_ds)
            plot_comparative_boxplots(all_point_data, param, locations, output_dir, master_ds)

    # 4. Roses
    if WINDROSE_AVAILABLE:
        for loc_name in locations.keys():
            plot_wave_rose(all_point_data.get(loc_name), loc_name, output_dir, master_ds)
    
    master_ds.close()

def main():
    print(f"\n🚀 Step 8: POI Wave Climate Analysis...")
    for region_name, cfg in REGIONS_CONFIG.items():
        process_region(region_name, cfg)
    print(f"\n✨ Step 8 Completed.")

if __name__ == "__main__":
    main()