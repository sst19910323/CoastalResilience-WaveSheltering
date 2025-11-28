# filename: 3_analyze_wave.py
# description: (Refactored) Calculates spatial statistics and generates
#              high-quality Cartopy plots for the manuscript.

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. 引入路径配置 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# --- 2. 绘图配置升级 (PAPER QUALITY) ---
PAPER_FONT_SIZE = 16
plt.rcParams.update({
    'font.size': PAPER_FONT_SIZE,
    'axes.labelsize': PAPER_FONT_SIZE + 2,
    'axes.titlesize': PAPER_FONT_SIZE + 4,
    'xtick.labelsize': PAPER_FONT_SIZE,
    'ytick.labelsize': PAPER_FONT_SIZE,
    'legend.fontsize': PAPER_FONT_SIZE,
    'figure.titlesize': PAPER_FONT_SIZE + 6,
    'figure.figsize': (12, 10), # 默认大尺寸
    'savefig.dpi': 300 # 高清输出
})

# --- 3. 区域配置 ---
# 只需要定义文件名和绘图参数，路径由 config 提供
REGIONS_CONFIG = {
    "Japan": {
        "nc_file": 'japan.nc',
        "plot_extent": [128, 148, 30, 46], 
        "quiver_scale": 50
    },
    "New_York": {
        "nc_file": 'new_york.nc',
        "plot_extent": [-77, -72, 37.5, 41.5],
        "quiver_scale": 40
    },
    "Melbourne": {
        "nc_file": 'melbourne.nc',
        "plot_extent": [143.5, 145.5, -39, -37.5],
        "quiver_scale": 20
    },
    "Guangzhou_Bay_Area": {
        "nc_file": 'guangzhou_bay.nc',
        "plot_extent": [113.5, 115.5, 21.8, 23.0],
        "quiver_scale": 20
    },
    "Hangzhou_Bay": {
        "nc_file": 'hangzhou_bay.nc',
        "plot_extent": [121.0, 123.0, 29.5, 31.8],
        "quiver_scale": 30
    }
}

# --- 4. 辅助函数 ---
LAT_COORD = 'latitude'; LON_COORD = 'longitude'; TIME_COORD = 'time'
VHM0_VAR = 'VHM0'; VTPK_VAR = 'VTPK'; VMDR_VAR = 'VMDR'
QUIVER_DENSITY = 0.05

def create_output_directory(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def plot_spatial_scalar_data(data_array, title_prefix, units, output_path, cmap='viridis', extent=None):
    if data_array is None: return
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plotting
    im = data_array.plot(ax=ax, x=LON_COORD, y=LAT_COORD, transform=ccrs.PlateCarree(),
                       cmap=cmap, 
                       add_colorbar=False) # 关掉自动 colorbar，手动加大的
    
    # Manual Colorbar (Bigger)
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(f'{title_prefix} ({units})', fontsize=PAPER_FONT_SIZE+2)
    cbar.ax.tick_params(labelsize=PAPER_FONT_SIZE)

    # Features
    ax.coastlines('10m', linewidth=1.5) # 加粗海岸线
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Gridlines (Bigger text)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': PAPER_FONT_SIZE, 'color': 'black'}
    gl.ylabel_style = {'size': PAPER_FONT_SIZE, 'color': 'black'}
    
    plt.title(f"{title_prefix}", fontsize=PAPER_FONT_SIZE+4, pad=20)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Saved: {os.path.basename(output_path)}")

def plot_spatial_wave_vectors(strength, direction, title_prefix, units, output_path, extent=None, scale=50, density=0.05):
    if strength is None or direction is None: return
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    lons, lats = strength[LON_COORD].values, strength[LAT_COORD].values
    dir_rad = np.deg2rad(direction.values)
    u = strength.values * np.sin(dir_rad)
    v = strength.values * np.cos(dir_rad)

    # Plot Background
    im = strength.plot(ax=ax, x=LON_COORD, y=LAT_COORD, transform=ccrs.PlateCarree(),
                     cmap='viridis', add_colorbar=False)
    
    # Manual Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label(f'{strength.name} ({units})', fontsize=PAPER_FONT_SIZE+2)
    cbar.ax.tick_params(labelsize=PAPER_FONT_SIZE)

    # Quiver
    step = max(1, int(len(lons) * density))
    ax.quiver(lons[::step], lats[::step], u[::step, ::step], v[::step, ::step],
              transform=ccrs.PlateCarree(), scale=scale, color='black', pivot='middle', width=0.003) # 加宽箭头

    # Features
    ax.coastlines('10m', linewidth=1.5)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    
    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': PAPER_FONT_SIZE, 'color': 'black'}
    gl.ylabel_style = {'size': PAPER_FONT_SIZE, 'color': 'black'}

    plt.title(f"{title_prefix}", fontsize=PAPER_FONT_SIZE+4, pad=20)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   Saved: {os.path.basename(output_path)}")

# --- 5. 区域处理流程 ---
def process_region(region_name, cfg):
    # 路径构造
    file_path = os.path.join(config.NC_DIR, cfg["nc_file"])
    
    # 依然输出到 raster_geometry_analysis_output/[Region] 下，方便查看
    output_dir = os.path.join(config.OUTPUT_ROOT, 'raster_geometry_analysis_output', region_name)
    
    plot_extent, quiver_scale = cfg["plot_extent"], cfg["quiver_scale"]

    if not os.path.exists(file_path):
        print(f"⚠️ SKIPPING: Input file not found -> {file_path}")
        return
        
    create_output_directory(output_dir)
    
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return

    print(f"   Calculating stats for {region_name}...")
    
    # 计算统计量
    mean_vhm0 = ds[VHM0_VAR].mean(dim=TIME_COORD, skipna=True)
    p99_vhm0 = ds[VHM0_VAR].quantile(0.99, dim=TIME_COORD, skipna=True)
    mean_vtpk = ds[VTPK_VAR].mean(dim=TIME_COORD, skipna=True)
    mean_vmdr = ds[VMDR_VAR].mean(dim=TIME_COORD, skipna=True)
    
    vhm0_units = ds[VHM0_VAR].attrs.get('units', 'm')
    vtpk_units = ds[VTPK_VAR].attrs.get('units', 's')

    # 绘图
    print("   Generating plots...")
    plot_spatial_scalar_data(mean_vhm0, "Mean VHM0", vhm0_units,
                               os.path.join(output_dir, "spatial_mean_vhm0.png"),
                               cmap='Spectral_r', extent=plot_extent)

    plot_spatial_scalar_data(p99_vhm0, "P99 VHM0", vhm0_units,
                               os.path.join(output_dir, "spatial_p99_vhm0.png"),
                               cmap='Spectral_r', extent=plot_extent)
    
    plot_spatial_scalar_data(mean_vtpk, "Mean VTPK", vtpk_units,
                               os.path.join(output_dir, "spatial_mean_vtpk.png"),
                               cmap='ocean_r', extent=plot_extent)

    plot_spatial_wave_vectors(mean_vhm0, mean_vmdr, "Mean VHM0 and Direction", vhm0_units,
                              os.path.join(output_dir, "spatial_mean_wave_vectors.png"),
                              extent=plot_extent, scale=quiver_scale, density=QUIVER_DENSITY)

    ds.close()

# --- 6. 主程序 ---
def main():
    print(f"\n🚀 Step 3: Spatial Wave Climate Analysis...")
    for region_name, cfg in REGIONS_CONFIG.items():
        print(f"\n--- Region: {region_name} ---")
        process_region(region_name, cfg)
    print(f"\n✨ Step 3 Completed.")

if __name__ == "__main__":
    main()