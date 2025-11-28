# filename: inspect_points_interactive.py
# description: [EXPERIMENTAL] Interactive tool using Plotly to inspect 
#              time-series data from specific NetCDF files. 
#              Used for preliminary data quality checks.

import netCDF4 as nc
import numpy as np
import plotly.graph_objects as go
import os

def read_nc(file_path):
    """Reads Lat/Lon, Time, and Variables from a NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            # Assumes single point data
            latitude = dataset.variables['latitude'][0]
            longitude = dataset.variables['longitude'][0]
            time = dataset.variables['time'][:].tolist()
            
            variables = {}
            for var in dataset.variables:
                if var not in ['latitude', 'longitude', 'time']:
                    variables[var] = dataset.variables[var][:]
                    
        return longitude, latitude, time, variables
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None

def compare_variable_across_files(file_list, variable_name, arrow_variable_name):
    """Generates an interactive Plotly chart comparing variables."""
    fig = go.Figure()

    for i, file_path in enumerate(file_list):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        _, _, time, variables = read_nc(file_path)
        if variables is None: continue

        var_data = np.squeeze(variables[variable_name])
        arrow_var_data = np.squeeze(variables[arrow_variable_name])

        # Main Line
        fig.add_trace(go.Scatter(
            x=np.array(time),
            y=var_data,
            mode='lines',
            name=f'File {i + 1}: {os.path.basename(file_path)}'
        ))

        # Arrows (Subsampled)
        sample_rate = 10
        if len(time) > sample_rate:
            arrow_times = time[::sample_rate]
            sampled_var = var_data[::sample_rate]
            sampled_angle = arrow_var_data[::sample_rate]
            
            # Plotly doesn't support vector fields natively easily on 2D lines,
            # using markers to represent direction is a good workaround
            fig.add_trace(go.Scatter(
                x=arrow_times,
                y=sampled_var,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, angleref='previous', angle=sampled_angle, color='red'),
                name=f'Direction {i + 1}',
                showlegend=False
            ))

    fig.update_layout(title=f"Comparison of {variable_name}", yaxis_title=variable_name, xaxis_title="Time Step")
    fig.show()

if __name__ == "__main__":
    # Example Usage: Replace with actual paths to test
    # files = ["/path/to/file1.nc", "/path/to/file2.nc"]
    # compare_variable_across_files(files, 'VHM0', 'VMDR')
    print("This is an interactive inspection tool. Please configure file paths in the script to run.")