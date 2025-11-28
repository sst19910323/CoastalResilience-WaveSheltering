import os

# --- 1. 定位项目根目录 ---
# 假设 config.py 在 src/ 下，往上跳一级就是项目根目录
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# --- 2. 定义数据输入路径 ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 原始 NC 文件路径
NC_DIR = os.path.join(DATA_DIR, 'raw_nc')

# TMS 切片路径
TMS_DIR = os.path.join(DATA_DIR, 'tms_tiles')

# 参考底图 TIFF 路径
TIFF_DIR = os.path.join(DATA_DIR, 'ref_tiffs')

# --- 3. 定义输出路径 (自动创建，防止报错) ---
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')

# 你原来的各种输出文件夹，统一归类到 outputs 下
EFFECTIVE_WAVE_DIR = os.path.join(OUTPUT_ROOT, 'effective_wave_outputs')
NEIGHBORHOOD_DIR = os.path.join(OUTPUT_ROOT, 'neighborhood_analysis_outputs')
FINAL_COMBINED_DIR = os.path.join(OUTPUT_ROOT, 'final_combined_outputs')
FIGURES_DIR = os.path.join(OUTPUT_ROOT, 'paper_figures') # 专门放论文大图

# 自动创建文件夹函数
def ensure_dirs():
    for d in [EFFECTIVE_WAVE_DIR, NEIGHBORHOOD_DIR, FINAL_COMBINED_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

# 执行创建
ensure_dirs()