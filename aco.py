import pygame
import numpy as np
import sys
import time
import random
import pygame_gui

pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BG_COLOR = (240, 240, 240)
CITY_COLOR = (0, 100, 200)
ANT_COLOR = (200, 0, 0)
PATH_COLOR = (150, 150, 150)
PHEROMONE_COLOR = (0, 100, 0)
TEXT_COLOR = (50, 50, 50)
FONT_SIZE = 16

class AntColonyOptimization:
    def __init__(self, coords, n_ants=10, decay=0.95, alpha=1.0, beta=2.0):
        self.coords = coords
        self.n_cities = len(coords)
        self.distances = self.calculate_distances()
        self.pheromones = np.ones((self.n_cities, self.n_cities)) / self.n_cities
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.best_path = None
        self.best_path_length = float('inf')
        self.iteration_best_paths = []
        self.iteration_best_lengths = []
        
        self.ant_positions = []  
        self.ant_trails = []     
        self.current_iteration = 0
        
    def calculate_distances(self):
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    distances[i, j] = np.sqrt((self.coords[i][0] - self.coords[j][0])**2 + 
                                             (self.coords[i][1] - self.coords[j][1])**2)
                else:
                    distances[i, j] = np.inf
        return distances
    
    def run_iteration(self):
        if self.n_cities < 2:
            return None, float('inf')  
            
        self.ant_positions = [np.random.randint(0, self.n_cities) for _ in range(self.n_ants)]
        self.ant_trails = [[pos] for pos in self.ant_positions]
        
        all_paths = []
        
        for ant in range(self.n_ants):
            path = self.gen_path(self.ant_positions[ant])
            all_paths.append(path)
            
        self.spread_pheromone(all_paths)
        self.pheromones = self.pheromones * self.decay
        
        iteration_best_path = min(all_paths, key=lambda x: self.path_length(x))
        iteration_best_length = self.path_length(iteration_best_path)
        self.iteration_best_paths.append(iteration_best_path)
        self.iteration_best_lengths.append(iteration_best_length)
        
        if iteration_best_length < self.best_path_length:
            self.best_path = iteration_best_path
            self.best_path_length = iteration_best_length
            
        self.current_iteration += 1
        return iteration_best_path, iteration_best_length
    
    def gen_path(self, start):
        path = [start]
        visited = set([start])
        
        while len(visited) < self.n_cities:
            current = path[-1]
            unvisited = list(set(range(self.n_cities)) - visited)
            
            probabilities = self.calculate_probabilities(current, unvisited)
            
            next_city = np.random.choice(unvisited, p=probabilities)
            
            path.append(next_city)
            visited.add(next_city)
            
            ant_ind = self.ant_positions.index(start)
            self.ant_trails[ant_ind].append(next_city)
            
        path.append(path[0])
        
        ant_ind = self.ant_positions.index(start)
        self.ant_trails[ant_ind].append(path[0])
        
        return path
    
    def calculate_probabilities(self, current, unvisited):
        probabilities = []
        denominator = 0
        
        for city in unvisited:
            pheromone = self.pheromones[current, city]
            distance = self.distances[current, city]
            numerator = pheromone**self.alpha * (1.0/distance)**self.beta
            denominator += numerator
            probabilities.append(numerator)
        
        if denominator == 0:
            return np.ones(len(unvisited)) / len(unvisited)
        probabilities = np.array(probabilities) / denominator
        return probabilities
    
    def spread_pheromone(self, all_paths):
        for path in all_paths:
            path_length = self.path_length(path)
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i+1]] += 1.0 / path_length
    
    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distances[path[i], path[i+1]]
        return length
    
    def update_coords(self, coords):
        self.coords = coords
        self.n_cities = len(coords)
        self.distances = self.calculate_distances()
        self.pheromones = np.ones((self.n_cities, self.n_cities)) / self.n_cities
        self.best_path = None
        self.best_path_length = float('inf')
        self.iteration_best_paths = []
        self.iteration_best_lengths = []
        self.current_iteration = 0
    
    def get_ant_positions(self):
        return self.ant_positions
    
    def get_ant_trails(self):
        return self.ant_trails
    
    def get_best_path(self):
        return self.best_path
    
    def get_best_path_length(self):
        return self.best_path_length
    
    def get_pheromone_levels(self):
        return self.pheromones

class ACOVisualizer:
    def __init__(self, n_cities=20, n_ants=10, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.n_cities = n_cities
        self.n_ants = n_ants
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ant Colony Optimization Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        
        self.ui_manager = pygame_gui.UIManager((width, height))
        
        self.cities = []
        self.generate_cities()
        
        self.aco = AntColonyOptimization(
            self.cities, 
            n_ants=n_ants, 
            decay=0.95, 
            alpha=1.0, 
            beta=2.0
        )
        
        self.paused = True
        self.speed = 10
        self.step_mode = False
        self.show_pheromones = True
        self.target_iteration = 100
        self.running_to_target = False
        
        self.manual_mode = False
        
        self.create_ui()
        
    def generate_cities(self):
        self.cities = []
        
        padding = 100
        city_area_width = self.width - 300 - padding * 2
        city_area_height = self.height - padding * 2
        
        for _ in range(self.n_cities):
            x = random.randint(padding, padding + city_area_width)
            y = random.randint(padding, padding + city_area_height)
            self.cities.append((x, y))
    
    def add_city(self, pos):
        if pos[0] < self.width - 300:
            self.cities.append(pos)
            self.n_cities = len(self.cities)
            self.param_sliders['cities'].set_current_value(self.n_cities)
            self.aco.update_coords(self.cities)
    
    def clear_cities(self):
        self.cities = []
        self.n_cities = 0
        self.aco.update_coords(self.cities)
    
    def create_ui(self):
        panel_rect = pygame.Rect(self.width - 280, 0, 280, self.height)
        
        button_width = 120
        button_height = 30
        button_margin = 10
        
        y_pos = 20
        
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, button_width, button_height),
            text='Start',
            manager=self.ui_manager
        )
        
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + button_width + 20, y_pos, button_width, button_height),
            text='Pause',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin
        
        self.load_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, button_width, button_height),
            text='Load Cities',
            manager=self.ui_manager
        )
        
        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + button_width + 20, y_pos, button_width, button_height),
            text='Save Cities',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin
        
        self.step_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, button_width, button_height),
            text='Step',
            manager=self.ui_manager
        )
        
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + button_width + 20, y_pos, button_width, button_height),
            text='Reset',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin
        
        self.manual_mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, button_width, button_height),
            text='Add Cities Manually',
            manager=self.ui_manager
        )
        
        self.clear_cities_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + button_width + 20, y_pos, button_width, button_height),
            text='Clear Cities',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin * 2
        
        self.speed_slider_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
            text='Animation Speed',
            manager=self.ui_manager
        )
        
        y_pos += 25
        
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
            start_value=self.speed,
            value_range=(1, 60),
            manager=self.ui_manager
        )
        
        y_pos += 30 + button_margin
        
        self.param_labels = {}
        self.param_sliders = {}
        
        params = {
            'cities': {'label': 'Cities', 'value': self.n_cities, 'range': (5, 50)},
            'ants': {'label': 'Ants', 'value': self.n_ants, 'range': (5, 100)},
            'alpha': {'label': 'Alpha (Pheromone Weight)', 'value': 1.0, 'range': (0.1, 5.0)},
            'beta': {'label': 'Beta (Distance Weight)', 'value': 2.0, 'range': (0.1, 10.0)},
            'decay': {'label': 'Pheromone Decay', 'value': 0.95, 'range': (0.5, 0.99)}
        }
        
        for param, details in params.items():
            self.param_labels[param] = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
                text=details['label'],
                manager=self.ui_manager
            )
            
            y_pos += 25
            
            self.param_sliders[param] = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
                start_value=details['value'],
                value_range=details['range'],
                manager=self.ui_manager
            )
            
            y_pos += 30 + button_margin
        
        self.run_to_target_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, button_width, button_height),
            text='Run to Target',
            manager=self.ui_manager
        )
        
        self.reset_iteration_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + button_width + 20, y_pos, button_width, button_height),
            text='Reset Iteration',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin * 2
        
        self.target_iter_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
            text='Target Iteration',
            manager=self.ui_manager
        )
        
        y_pos += 25
        
        self.target_iter_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
            start_value=100,
            value_range=(1, 1000),
            manager=self.ui_manager
        )
        
        y_pos += 30 + button_margin
        
        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 60, y_pos, 150, button_height),
            text='Apply Parameters',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin * 2
        
        self.show_pheromones_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_rect.left + 60, y_pos, 150, button_height),
            text='Toggle Pheromones',
            manager=self.ui_manager
        )
        
        y_pos += button_height + button_margin * 3
        
        self.mode_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 20),
            text='Current Mode: Normal',
            manager=self.ui_manager
        )
        
        y_pos += 25
        
        self.instructions_label = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect(panel_rect.left + 10, y_pos, 250, 100),
            html_text="<b>Instructions:</b><br>"
                    "- Click 'Add Cities Manually' to place cities<br>"
                    "- Click anywhere to add a city<br>"
                    "- Press 'Apply Parameters' when done<br>"
                    "- Use 'Start/Pause/Step' to control simulation",
            manager=self.ui_manager
        )
    
    def apply_parameters(self):
        n_cities = int(self.param_sliders['cities'].get_current_value())
        n_ants = int(self.param_sliders['ants'].get_current_value())
        alpha = self.param_sliders['alpha'].get_current_value()
        beta = self.param_sliders['beta'].get_current_value()
        decay = self.param_sliders['decay'].get_current_value()
        
        if n_cities != self.n_cities and not self.manual_mode:
            self.n_cities = n_cities
            self.generate_cities()
        
        self.aco = AntColonyOptimization(
            self.cities, 
            n_ants=n_ants, 
            decay=decay, 
            alpha=alpha, 
            beta=beta
        )
        
        self.n_ants = n_ants
        self.paused = True
    
    def draw(self):
        self.screen.fill(BG_COLOR)
        
        panel_rect = pygame.Rect(self.width - 280, 0, 280, self.height)
        pygame.draw.rect(self.screen, (220, 220, 220), panel_rect)
        
        for i, (x, y) in enumerate(self.cities):
            pygame.draw.circle(self.screen, CITY_COLOR, (x, y), 8)
            label = self.font.render(str(i), True, TEXT_COLOR)
            self.screen.blit(label, (x + 10, y - 10))
        
        if self.show_pheromones and self.n_cities > 1:
            pheromones = self.aco.get_pheromone_levels()
            max_pheromone = np.max(pheromones) if np.max(pheromones) > 0 else 1
            
            for i in range(self.n_cities):
                for j in range(i+1, self.n_cities):
                    if pheromones[i, j] > 0:
                        level = max(pheromones[i, j], pheromones[j, i])
                        norm_level = level / max_pheromone
                        width = int(1 + 5 * norm_level)
                        alpha = int(50 + 200 * norm_level)
                        
                        line_color = (*PHEROMONE_COLOR, alpha)
                        
                        pygame.draw.line(self.screen, line_color, self.cities[i], self.cities[j], width)
        
        if self.n_cities > 1:
            for trail in self.aco.get_ant_trails():
                if len(trail) > 1:
                    points = [self.cities[city] for city in trail]
                    pygame.draw.lines(self.screen, PATH_COLOR, False, points, 1)
        
        if self.n_cities > 1:
            for ant_ind, pos in enumerate(self.aco.get_ant_positions()):
                if ant_ind < len(self.aco.get_ant_trails()) and self.aco.get_ant_trails()[ant_ind]:
                    trail = self.aco.get_ant_trails()[ant_ind]
                    if trail:
                        last_pos = trail[-1]
                        ant_x, ant_y = self.cities[last_pos]
                        pygame.draw.circle(self.screen, ANT_COLOR, (ant_x, ant_y), 5)
        
        best_path = self.aco.get_best_path()
        if best_path:
            points = [self.cities[city] for city in best_path]
            pygame.draw.lines(self.screen, (0, 0, 0), True, points, 2)
        
        info_text = [
            f"Iteration: {self.aco.current_iteration} / {int(self.target_iter_slider.get_current_value())}",
            f"Best path length: {self.aco.get_best_path_length():.2f}" if self.aco.get_best_path_length() != float('inf') else "Best path length: N/A",
            f"Ants: {self.n_ants}",
            f"Cities: {self.n_cities}",
            f"Speed: {self.speed} FPS",
            f"Mode: {'Manual City Placement' if self.manual_mode else 'Normal'}"
        ]
        
        info_y = 10
        for text in info_text:
            label = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(label, (10, info_y))
            info_y += 25
        
        self.mode_label.set_text(f"Current Mode: {'Manual City Placement' if self.manual_mode else 'Normal'}")
        
        self.ui_manager.draw_ui(self.screen)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        
        while running:
            time_delta = self.clock.tick(self.speed) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.manual_mode:
                        self.add_city(event.pos)
                
                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.start_button:
                            self.paused = False
                            self.manual_mode = False
                        elif event.ui_element == self.pause_button:
                            self.paused = True
                        elif event.ui_element == self.step_button:
                            self.aco.run_iteration()
                        elif event.ui_element == self.reset_button:
                            self.apply_parameters()
                        elif event.ui_element == self.apply_button:
                            self.manual_mode = False
                            self.apply_parameters()
                        elif event.ui_element == self.show_pheromones_button:
                            self.show_pheromones = not self.show_pheromones
                        elif event.ui_element == self.manual_mode_button:
                            self.manual_mode = not self.manual_mode
                            self.paused = True
                        elif event.ui_element == self.clear_cities_button:
                            self.clear_cities()
                            self.param_sliders['cities'].set_current_value(0)
                        elif event.ui_element == self.load_button:
                            self.load_cities_dialog()
                        elif event.ui_element == self.save_button:
                            self.save_cities_dialog()
                        elif event.ui_element == self.run_to_target_button:
                            self.run_to_target()
                        elif event.ui_element == self.reset_iteration_button:
                            self.aco.current_iteration = 0
                            self.aco.iteration_best_paths = []
                            self.aco.iteration_best_lengths = []

                
                elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.speed_slider:
                        self.speed = int(event.value)
                
                self.ui_manager.process_events(event)
            
            if not self.paused and not self.manual_mode and self.n_cities > 1:
                self.aco.run_iteration()
                
                if self.running_to_target:
                    target = int(self.target_iter_slider.get_current_value())
                    if self.aco.current_iteration >= target:
                        self.running_to_target = False
                        self.paused = True
            
            self.ui_manager.update(time_delta)
            
            self.draw()
        
        pygame.quit()
    
    def load_cities_dialog(self):
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Load Cities",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            cities = load_cities_from_file(file_path)
            if cities:
                self.cities = cities
                self.n_cities = len(cities)
                self.aco.update_coords(cities)
                self.param_sliders['cities'].set_current_value(len(cities))
                print(f"Loaded {len(cities)} cities from {file_path}")
                
                self.paused = True
    
    def save_cities_dialog(self):
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            title="Save Cities",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".txt"
        )
        
        if file_path:
            success = save_cities_to_file(self.cities, file_path)
            if success:
                print(f"Saved {len(self.cities)} cities to {file_path}")
            else:
                print("Failed to save cities")
    
    def run_to_target(self):
        if self.n_cities < 2:
            return
            
        target = int(self.target_iter_slider.get_current_value())
        current = self.aco.current_iteration
        
        if current >= target:
            return
            
        self.running_to_target = True
        self.paused = False
        self.manual_mode = False

def load_cities_from_file(filename):
    try:
        cities = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        cities.append((x, y))
                    except ValueError:
                        print(f"Warning: Could not parse line: {line}")
        return cities
    except Exception as e:
        print(f"Error loading cities from file: {e}")
        return None

def save_cities_to_file(cities, filename):
    try:
        with open(filename, 'w') as f:
            for x, y in cities:
                f.write(f"{x},{y}\n")
        return True
    except Exception as e:
        print(f"Error saving cities to file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ant Colony Optimization Visualization")
    parser.add_argument("--cities", type=int, default=20, help="Number of cities")
    parser.add_argument("--ants", type=int, default=20, help="Number of ants")
    parser.add_argument("--load", type=str, help="Load cities from file")
    parser.add_argument("--save", type=str, help="Save cities to file")
    
    args = parser.parse_args()
    
    visualizer = ACOVisualizer(n_cities=args.cities, n_ants=args.ants)
    
    if args.load:
        cities = load_cities_from_file(args.load)
        if cities:
            visualizer.cities = cities
            visualizer.n_cities = len(cities)
            visualizer.aco.update_coords(cities)
            visualizer.param_sliders['cities'].set_current_value(len(cities))
    
    visualizer.run()
    
    if args.save:
        save_cities_to_file(visualizer.cities, args.save)
