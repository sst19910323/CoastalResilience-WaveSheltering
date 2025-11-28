# filename: main.py
# description: MASTER EXECUTION SCRIPT.
#              Sequentially runs the entire analysis pipeline from Step 1 to Step 8.
#              Usage: python main.py

import subprocess
import os
import sys
import time

# 定义脚本执行的绝对顺序
PIPELINE_STEPS = [
    # --- Part 1: Geography & Preprocessing ---
    "src/1_analyze_coastline.py",       # Extract coastline, calculate direction/curvature
    "src/2_analyze_geometry.py",        # Generate buffers and propagate geometry attributes
    
    # --- Part 2: Wave Data Processing ---
    "src/3_analyze_wave.py",            # Calculate raw wave statistics and generate input plots
    "src/4_process_wave_data.py",       # Align/Reproject NC data to match GeoTIFF grids
    
    # --- Part 3: Spatial Modeling ---
    "src/5_model_a_propagation.py",     # Run Model A (Line-Source Propagation)
    "src/6_model_b_neighborhood.py",    # Run Model B (Neighborhood Average)
    "src/7_model_fusion.py",            # Run Final Fusion (Model A + B)
    
    # --- Part 4: Statistical Analysis ---
    "src/analysis/analyze_poi.py"     # Generate Time-Series, Boxplots, and Wave Roses (Figs 6&7)
]

def run_script(script_relative_path):
    """Executes a python script as a subprocess."""
    
    # 获取脚本的绝对路径，兼容不同操作系统
    script_path = os.path.abspath(script_relative_path)
    
    if not os.path.exists(script_path):
        print(f"❌ CRITICAL ERROR: Script not found at {script_path}")
        return False

    print(f"\n" + "="*60)
    print(f"▶️  RUNNING: {os.path.basename(script_relative_path)}")
    print(f"    Path: {script_path}")
    print("="*60 + "\n")

    start_time = time.time()
    
    # 使用当前的 Python 解释器执行子脚本
    # check=True 会在脚本返回非零状态码时抛出异常
    try:
        subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start_time
        print(f"\n✅ FINISHED: {os.path.basename(script_relative_path)} in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {os.path.basename(script_relative_path)} exited with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ FAILED: An unexpected error occurred: {e}")
        return False

def main():
    print(f"🚀 Starting Coastal Resilience Analysis Pipeline")
    print(f"📂 Working Directory: {os.getcwd()}")
    print(f"📜 Total Steps: {len(PIPELINE_STEPS)}\n")

    total_start = time.time()
    
    for i, script in enumerate(PIPELINE_STEPS, 1):
        print(f"\n[Step {i}/{len(PIPELINE_STEPS)}]")
        success = run_script(script)
        
        if not success:
            print("\n⛔ PIPELINE ABORTED due to error in previous step.")
            sys.exit(1)
            
    total_elapsed = time.time() - total_start
    print(f"\n" + "="*60)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total Time: {total_elapsed/60:.1f} minutes")
    print(f"📂 Check the 'outputs/' directory for results.")
    print("="*60)

if __name__ == "__main__":
    main()