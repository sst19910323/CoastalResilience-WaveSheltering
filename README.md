# Inherent Coastal Resilience: Wave Sheltering as a Primary Determinant of Global Urbanization Patterns

**Code Repository for the Manuscript Submitted to *International Journal of Applied Earth Observation and Geoinformation (JAG)***

This repository contains the source code, spatial models, and analysis scripts for the study **"Inherent Coastal Resilience"**. It provides a fully reproducible pipeline to quantify the inland influence of wave energy and correlate it with global settlement patterns.

---

## 📂 Repository Structure

The project is organized to ensure modularity and reproducibility.

```text
CoastalResilience-WaveSheltering/
├── data/
│   ├── raw_nc/               # [User Action Required] Place raw CMEMS NetCDF files here
│   ├── tms_tiles/            # Downloaded map tiles for basemaps
│   └── ref_tiffs/            # Mosaicked GeoTIFFs used as reference grids
├── src/
│   ├── config.py             # Global path configuration
│   ├── 1_analyze_coastline.py    # Step 1: Extracts coastline geometry (direction/curvature)
│   ├── 2_analyze_geometry.py     # Step 2: Generates buffer zones and propagates geometry
│   ├── 3_analyze_wave.py         # Step 3: Calculates spatial wave statistics
│   ├── 4_process_wave_data.py    # Step 4: Aligns/Reprojects NC data to high-res grids
│   ├── 5_model_a_propagation.py  # Step 5: Runs Model A (Line-Source Propagation)
│   ├── 6_model_b_neighborhood.py # Step 6: Runs Model B (Neighborhood Average)
│   ├── 7_model_fusion.py         # Step 7: Fuses models (Model A + B)
│   ├── analysis/
│   │   └── analyze_poi.py        # Step 8: Point-of-Interest Analysis (Figures 6 & 7)
│   └── utils/                # Helper scripts (Downloader, Mosaicking)
├── experimental/             # Documented "failed" attempts & exploratory code (Transparency)
├── outputs/                  # All generated maps, figures, and intermediate files
├── main.py                   # Master script to run the full workflow
├── requirements.txt          # Python dependencies
└── README.md

🛠️ Installation & Requirements
Clone the repository:

git clone [https://github.com/sst19910323/CoastalResilience-WaveSheltering.git](https://github.com/sst19910323/CoastalResilience-WaveSheltering.git)
cd CoastalResilience-WaveSheltering
Install dependencies: It is recommended to use a virtual environment (conda or venv).


pip install -r requirements.txt
Key Dependencies: xarray, netCDF4, rasterio, numpy, scipy, matplotlib, cartopy, opencv-python.

📦 Data Acquisition
Due to the significant file size (>2GB total), the raw global wave reanalysis data cannot be hosted directly on GitHub. To reproduce the analysis from scratch:

Wave Data Source:

Product: Copernicus Marine Service (CMEMS) - GLOBAL_ANALYSISFORECAST_WAV_001_027

DOI: 10.48670/moi-00017

Action: Download the subsets for the 5 study regions covering Jan 2021 - Jan 2025.

Naming Convention: Place files in data/raw_nc/ and rename to: japan.nc, new_york.nc, melbourne.nc, guangzhou_bay.nc, hangzhou_bay.nc.

Basemaps:

Run python src/utils/tms_downloader.py to acquire map tiles.

Run python src/utils/tms_mosaic_tool.py to generate reference GeoTIFFs.

🚀 Reproduction Guide
You can run the entire analysis pipeline using the master script:

Bash

python main.py
This script will sequentially execute Steps 1 through 8, printing progress to the console. All results will be generated in the outputs/ directory.

🧪 Experimental & Negative Results
In the spirit of scientific transparency, we include the experimental/ directory containing code for approaches that were explored but ultimately discarded.

calculate_effective_wave_component.py: An attempt to calculate "Effective Wave Height" based on the cosine of the angle between wave direction and coastline normal. This physically-based approach was theoretically sound but failed to produce statistically robust results due to the coarse resolution of global wave direction vectors relative to complex coastline geometry.

experimental_fusion_attempt.py: An early attempt to fuse the effective wave component with the mean field.

We retained the robust scalar H_mean models (Model A & B) for the final manuscript.

📝 Citation
If you use this code or methodology, please cite our paper:

[Author Names]. (2025). Inherent Coastal Resilience: Wave Sheltering as a Primary Determinant of Global Urbanization Patterns. International Journal of Applied Earth Observation and Geoinformation (Submitted).

For any technical issues, please open an issue in this repository.