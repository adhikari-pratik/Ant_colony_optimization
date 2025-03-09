# Ant Colony Optimization (ACO) Visualization

This is a Python-based visualization tool for the Ant Colony Optimization algorithm applied to the Traveling Salesman Problem (TSP). The application provides an interactive GUI to visualize how ants find the shortest path connecting all cities.

## Features

- Interactive visualization of ACO algorithm in real-time
- Support for both Euclidean and geographic distances
- Adjustable parameters (alpha, beta, decay rate, etc.)
- Manual city placement or random generation
- Loading cities from CSV or JSON files
- Saving city configurations
- Step-by-step or continuous animation
- Pheromone trail visualization

## Installation

1. Clone the repository or download the source code
2. Install required packages:

```bash
pip install -r requirements.txt
```

## How to Run

### Basic Usage

```bash
python aco.py
```

### Command-line Options

```bash
python aco.py --cities 20 --ants 30 --load european_cities.csv --save output.csv
```

Parameters:
- `--cities`: Number of random cities to generate (default: 20)
- `--ants`: Number of ants to use (default: 20)
- `--load`: Load cities from a file
- `--save`: Save the final city configuration to a file

## Loading City Data

The application supports loading cities from CSV or JSON files.

### CSV Format

For geographic coordinates, the CSV should contain columns for:
- City name
- Latitude
- Longitude
- Country (optional)

Example with `european_cities.csv`:
```csv
city,lat,lng,country
London,51.5074,0.1278,UK
Paris,48.8566,2.3522,France
Berlin,52.5200,13.4050,Germany
...
```

To load this file:
1. Click the "Load Cities" button
2. Select your CSV file
3. The application will automatically detect geographic coordinates and calculate distances using the Haversine formula

### JSON Format

The application supports different JSON formats:
1. Geographic coordinates:
```json
[
  {"name": "London", "lat": 51.5074, "lng": 0.1278, "country": "UK"},
  {"name": "Paris", "lat": 48.8566, "lng": 2.3522, "country": "France"}
]
```

2. Screen coordinates:
```json
[
  {"x": 100, "y": 200},
  {"x": 300, "y": 400}
]
```

## User Interface Controls

### Buttons
- **Start/Pause**: Begin or pause the simulation
- **Step**: Run a single iteration of the ACO algorithm
- **Load Cities**: Load city data from a file
- **Save Cities**: Save current city configuration
- **Reset**: Reset the simulation with current parameters
- **Add Cities Manually**: Enter manual city placement mode
- **Clear Cities**: Remove all cities
- **Run to Target**: Run the simulation until reaching the target iteration
- **Reset Iteration**: Reset the iteration counter
- **Apply Parameters**: Apply any parameter changes
- **Toggle Pheromones**: Show or hide pheromone trails

### Sliders
- **Animation Speed**: Control the simulation speed (fps)
- **Cities**: Number of cities (when regenerating)
- **Ants**: Number of ants in the simulation
- **Alpha**: Pheromone weight in path selection
- **Beta**: Distance weight in path selection
- **Decay**: Pheromone evaporation rate
- **Target Iteration**: Set the target iteration for "Run to Target"

## Algorithm Parameters

- **Alpha (α)**: Controls the influence of pheromones on path selection. Higher values make ants more likely to follow paths with stronger pheromone trails.
- **Beta (β)**: Controls the influence of distance on path selection. Higher values make ants prefer shorter paths.
- **Decay Rate**: Controls how quickly pheromones evaporate. Lower values make the algorithm explore more.
- **Number of Ants**: More ants can explore more paths but may slow down the simulation.

## Tips for Use

1. Use "Add Cities Manually" to create a custom city configuration
2. For geographic data, load a CSV with latitude and longitude information
3. Adjust parameters to see how they affect the solution quality
4. Use "Step" to see the algorithm progress iteration by iteration
5. Watch the pheromone trails grow stronger on better paths
6. The best path found so far is shown in black

## Example with European Cities

1. Download the `european_cities.csv` file with major European cities
2. Run the program: `python aco.py`
3. Click "Load Cities" and select `european_cities.csv`
4. The cities will appear on the map with geographic scaling
5. Click "Apply Parameters" to initialize the ACO algorithm with these cities
6. Click "Start" to run the algorithm and find the shortest route

## Output

The best path found by the ACO algorithm will be displayed on the screen. You can save the city configuration and coordinates by clicking "Save Cities".

## Screenshots
![ACO Visualization Example](./screenshots/aco_example.png)

## Customization

You can modify the following constants in the code:
- `SCREEN_WIDTH` and `SCREEN_HEIGHT`: Change the window size
- Colors: `BG_COLOR`, `CITY_COLOR`, `ANT_COLOR`, etc.
- `FONT_SIZE`: Change the text size

## License

This project is provided for educational purposes.
