# Ant Colony Optimization for TSP

An interactive visualization tool for the Ant Colony Optimization algorithm solving the Traveling Salesman Problem.

## Features

- **Real-time Visualization**: Watch ants explore different paths between cities
- **Interactive Controls**: Start, pause, step through iterations, or run to a target iteration
- **Parameter Tuning**: Adjust number of cities, ants, pheromone influence (α), distance influence (β), and decay rate
- **City Management**: Generate random cities or manually place them by clicking
- **File Operations**: Save and load city configurations
- **Best Path Tracking**: Visualize the best path found so far and its length
- **Pheromone Visualization**: Toggle visibility of pheromone concentrations between cities

## Installation

### Requirements
- Python 3.6+
- NumPy
- Pygame
- Pygame-GUI

### Setup
```bash
# Install required dependencies
pip install numpy pygame pygame_gui
```

## Usage

### Basic Execution
```bash
python aco.py
```

### Command Line Arguments
```bash
python aco.py [--cities N] [--ants N] [--load FILENAME] [--save FILENAME]
```

Options:
- `--cities N`: Set initial number of cities (default: 20)
- `--ants N`: Set number of ants (default: 20)
- `--load FILENAME`: Load cities from file
- `--save FILENAME`: Save cities to file after execution

## User Interface

### Controls
- **Start**: Begin automatic simulation
- **Pause**: Pause simulation
- **Step**: Execute one iteration
- **Reset**: Reinitialize the simulation with current parameters
- **Add Cities Manually**: Enable manual city placement mode (click to add cities)
- **Clear Cities**: Remove all cities
- **Toggle Pheromones**: Show/hide pheromone trails
- **Run to Target**: Run simulation until reaching target iteration number
- **Reset Iteration**: Reset iteration counter to 0

### Parameters
Adjust these using sliders:
- **Cities**: Number of cities (5-50)
- **Ants**: Number of ants (5-100)
- **Alpha**: Pheromone influence weight (0.1-5.0)
- **Beta**: Distance influence weight (0.1-10.0)
- **Decay**: Pheromone evaporation rate (0.5-0.99)
- **Target Iteration**: Set target iteration number (1-1000)
- **Animation Speed**: Frames per second (1-60)

## Algorithm Details

Ant Colony Optimization mimics how ant colonies find optimal paths through pheromone communication:

1. Ants are randomly placed on cities
2. Each ant constructs a complete tour by selecting cities based on:
   - Pheromone levels on paths (α parameter controls influence)
   - Distance between cities (β parameter controls influence)
3. After all ants complete their tours:
   - Pheromones are deposited on all paths (inversely proportional to path length)
   - Existing pheromones evaporate according to decay rate
4. The process repeats, with paths to better solutions receiving more pheromones
5. Over time, the colony converges toward optimal or near-optimal solutions

## File Format

Cities can be saved and loaded using simple CSV format:

```
x,y
100,200
150,300
...
```

Each line represents coordinates of a city.

## Examples

### Standard Run
```bash
python aco.py --cities 25 --ants 30
```

### Load Saved Cities
```bash
python aco.py --load cities.txt
```

### Save Generated Solution
```bash
python aco.py --cities 20 --save solution.txt
```

## Visualization Details

- **Blue Circles**: Cities
- **Red Circles**: Ants
- **Black Line**: Best path found so far
- **Gray Lines**: Current ant trails
- **Green Lines**: Pheromone levels (darker = stronger)

## Tips for Best Results

- Start with fewer cities (10-20) to see algorithm behavior
- Increase alpha to emphasize pheromone following
- Increase beta to emphasize shorter distances
- Higher decay rates (0.9-0.99) give more stable convergence
- Lower decay rates (0.5-0.8) allow more exploration
