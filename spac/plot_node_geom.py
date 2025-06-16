import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

def read_station_data(filepath):
    """Reads station data from a file and returns it as a list of tuples."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as file:
        lines = file.readlines()

    title = lines[0].strip()
    data_lines = [line for line in lines[1:] if not line.startswith('#')]

    stations = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 4:
            print(f"Warning: Skipping malformed line: {line}")
            continue

        try:
            station_name = parts[0]
            longitude = float(parts[1])
            latitude = float(parts[2])
            elevation = float(parts[3])
            stations.append((station_name, longitude, latitude, elevation))
        except ValueError as e:
            print(f"Warning: Skipping line due to error: {line} Error: {e}")

    return title, stations

def format_lon_lat(value, pos):
    """Format latitude and longitude values for tick labels."""
    return f'{value:.6f}'

def plot_station_data(title, stations, filename):
    """Plots station data using matplotlib with improved quality and full lat/long values."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)  # Increase figure size and DPI
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability

    # Extract longitude and latitude ranges for setting tick formatting
    longitudes = [station[1] for station in stations]
    latitudes = [station[2] for station in stations]

    # Plot the stations
    for station_name, longitude, latitude, elevation in stations:
        ax.plot(longitude, latitude, marker='^', markersize=10, color='steelblue')

        # Add label with background and border for better visibility
        ax.text(
            longitude, latitude, station_name,
            fontsize=10,
            ha='center',
            va='bottom',
            bbox=dict(facecolor='none', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
            color='black'
        )

    # Set tick formatting to show full lat/lon values
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_lon_lat))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_lon_lat))

    # Set more frequent ticks on the axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks

    # Optionally set the ticks to cover the range of data points with more granularity
    lon_range = np.arange(min(longitudes), max(longitudes), 0.0001)  # Adjust interval as needed
    lat_range = np.arange(min(latitudes), max(latitudes), 0.0001)   # Adjust interval as needed

    ax.set_xticks(lon_range)
    ax.set_yticks(lat_range)

    # Rotate x-axis labels to vertical
    ax.tick_params(axis='x', labelrotation=90)

    # Save the figure
    fig.savefig(filename, bbox_inches='tight', dpi=300)  # Save with tight bounding box and high DPI

    plt.show()  # Display the plot


filepath = '/Users/keyser/Research/sz4d_ft/nodal_data/Enclave_1_station_data.txt' # Enter file name (.txt)
output_file = 'enclave1_plot.png'  # Output filename
try:
    title, stations = read_station_data(filepath)
    plot_station_data(title, stations, output_file)
except Exception as e:
    print(f"An error occurred: {e}")
