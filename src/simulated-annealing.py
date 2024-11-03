import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class MagicCube:
    def __init__(self, n):
        self.n = n
        self.cube = self.generate_random_cube()
        self.magic_number = self.calculate_magic_number()
        self.row_sums = np.sum(self.cube, axis=2)
        self.col_sums = np.sum(self.cube, axis=1)
        self.layer_sums = np.sum(self.cube, axis=0)

    def generate_random_cube(self):
        numbers = list(range(1, self.n**3 + 1))
        random.shuffle(numbers)
        cube = np.array(numbers).reshape(self.n, self.n, self.n)
        return cube

    def calculate_magic_number(self):
        return (self.n * (self.n**3 + 1)) // 2

    def swap(self, i1, j1, k1, i2, j2, k2):
        self.cube[i1, j1, k1], self.cube[i2, j2, k2] = self.cube[i2, j2, k2], self.cube[i1, j1, k1]
        self.update_sums(i1, j1, k1, i2, j2, k2)

    def update_sums(self, i1, j1, k1, i2, j2, k2):
        self.row_sums[i1, j1] = np.sum(self.cube[i1, j1, :])
        self.row_sums[i2, j2] = np.sum(self.cube[i2, j2, :])
        self.col_sums[i1, k1] = np.sum(self.cube[:, i1, k1])
        self.col_sums[i2, k2] = np.sum(self.cube[:, i2, k2])
        self.layer_sums[j1, k1] = np.sum(self.cube[i1, :, k1])
        self.layer_sums[j2, k2] = np.sum(self.cube[i2, :, k2])

    def calculate_objective_value(self):
        total_error = (np.abs(self.row_sums - self.magic_number).sum() +
                    np.abs(self.col_sums - self.magic_number).sum() +
                    np.abs(self.layer_sums - self.magic_number).sum())
        
        row_error_count = np.sum(np.abs(self.row_sums - self.magic_number) > 0)
        col_error_count = np.sum(np.abs(self.col_sums - self.magic_number) > 0)
        layer_error_count = np.sum(np.abs(self.layer_sums - self.magic_number) > 0)

        diagonal_error_count = 0

        for i in range(self.n):
            main_diagonal_sum = self.cube[i, range(self.n), range(self.n)].sum()
            anti_diagonal_sum = self.cube[i, range(self.n), range(self.n-1, -1, -1)].sum()
            vertical_diagonal_sum = self.cube[range(self.n), i, range(self.n)].sum()
            vertical_anti_diagonal_sum = self.cube[range(self.n), i, range(self.n-1, -1, -1)].sum()
            depth_diagonal_sum = self.cube[range(self.n), range(self.n), i].sum()
            depth_anti_diagonal_sum = self.cube[range(self.n), range(self.n-1, -1, -1), i].sum()

            main_diagonal_error = abs(main_diagonal_sum - self.magic_number)
            anti_diagonal_error = abs(anti_diagonal_sum - self.magic_number)
            vertical_diagonal_error = abs(vertical_diagonal_sum - self.magic_number)
            vertical_anti_diagonal_error = abs(vertical_anti_diagonal_sum - self.magic_number)
            depth_diagonal_error = abs(depth_diagonal_sum - self.magic_number)
            depth_anti_diagonal_error = abs(depth_anti_diagonal_sum - self.magic_number)

            if main_diagonal_error > 0:
                total_error += main_diagonal_error
                diagonal_error_count += 1
                
            if anti_diagonal_error > 0:
                total_error += anti_diagonal_error
                diagonal_error_count += 1
                
            if vertical_diagonal_error > 0:
                total_error += vertical_diagonal_error
                diagonal_error_count += 1
                
            if vertical_anti_diagonal_error > 0:
                total_error += vertical_anti_diagonal_error
                diagonal_error_count += 1
                
            if depth_diagonal_error > 0:
                total_error += depth_diagonal_error
                diagonal_error_count += 1
                
            if depth_anti_diagonal_error > 0:
                total_error += depth_anti_diagonal_error
                diagonal_error_count += 1

        ruang_diagonals = [
            self.cube[range(self.n), range(self.n), range(self.n)].sum(),
            self.cube[range(self.n), range(self.n), range(self.n-1, -1, -1)].sum(),
            self.cube[range(self.n), range(self.n-1, -1, -1), range(self.n)].sum(),
            self.cube[range(self.n), range(self.n-1, -1, -1), range(self.n-1, -1, -1)].sum()
        ]

        ruang_error_count = 0
        for ruang_diagonal_sum in ruang_diagonals:
            ruang_error = abs(ruang_diagonal_sum - self.magic_number)
            if ruang_error > 0:
                total_error += ruang_error
                ruang_error_count += 1

        total_error_count = row_error_count + col_error_count + layer_error_count + diagonal_error_count + ruang_error_count

        return total_error, total_error_count

    def copy(self):
        new_cube = MagicCube(self.n)
        new_cube.cube = np.copy(self.cube)
        return new_cube


class SimulatedAnnealing:
    def __init__(self, initial_cube, initial_temperature, cooling_rate, stopping_temperature, tolerance):
        self.cube = initial_cube
        self.current_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.tolerance = tolerance
        self.objective_history = []
        self.probability_history = []
        self.stuck_count = 0
        self.initial_state = np.copy(initial_cube.cube)
        self.initial_objective_value = initial_cube.calculate_objective_value()

    def acceptance_probability(self, old_cost, new_cost):
        old_total_error = old_cost[0]
        new_total_error = new_cost[0]
        
        if new_total_error < old_total_error:
            return 1.0
        else:
            return math.exp((old_total_error - new_total_error) / self.current_temperature)


    def get_neighbor(self):
        i1, j1, k1 = random.randint(0, self.cube.n - 1), random.randint(0, self.cube.n - 1), random.randint(0, self.cube.n - 1)
        i2, j2, k2 = random.randint(0, self.cube.n - 1), random.randint(0, self.cube.n - 1), random.randint(0, self.cube.n - 1)
        neighbor = self.cube.copy()
        neighbor.swap(i1, j1, k1, i2, j2, k2)
        return neighbor

    def anneal(self):
        start_time = time.time()
        
        current_solution = self.cube
        current_cost = current_solution.calculate_objective_value()
        best_solution = current_solution
        best_cost = current_cost

        print("Initial Cube State:\n", self.initial_state)
        print(f"Initial Objective Function Value (total_error, error_count): {self.initial_objective_value}\n")

        while self.current_temperature > self.stopping_temperature:
            neighbor = self.get_neighbor()
            new_cost = neighbor.calculate_objective_value()

            acceptance_prob = self.acceptance_probability(current_cost, new_cost)
            self.probability_history.append(acceptance_prob)

            if acceptance_prob == 1 or acceptance_prob > random.uniform(0, 1):
                current_solution = neighbor
                current_cost = new_cost
            else:
                self.stuck_count += 1

            if new_cost[1] < best_cost[1] or (new_cost[1] == best_cost[1] and new_cost[0] < best_cost[0]):
                best_solution = neighbor
                best_cost = new_cost

            self.objective_history.append(current_cost[0])

            self.current_temperature *= self.cooling_rate

            if best_cost[1] <= self.tolerance:
                break

        end_time = time.time()
        duration = end_time - start_time
        self.final_state = np.copy(best_solution.cube)
        final_objective_value = best_solution.calculate_objective_value()

        show_cube(best_solution)
        print("\nFinal Cube State:\n", self.final_state)
        print(f"\nFinal Objective Function Value (total_error, error_count): {final_objective_value}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Frequency of being 'stuck' at local optima: {self.stuck_count}")
        
        self.plot_results(duration, best_cost[0])
        
        return best_solution, best_cost, duration

    def plot_results(self, duration, best_cost):
        plt.figure(figsize=(12, 5))
        plt.plot(self.objective_history, label='Objective Value (total_error)')
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        plt.title("Objective Function Over Iterations")
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(self.probability_history, label='Acceptance Probability (e^ΔE/T)')
        plt.xlabel("Iterations")
        plt.ylabel("Acceptance Probability")
        plt.title("Acceptance Probability Over Iterations")
        plt.legend()
        plt.show()

def show_cube(cube_data):
    fig = plt.figure(figsize=(20, 50))
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(cube_data, MagicCube):
        cube = cube_data.cube
    else:
        cube = cube_data

    n = cube.shape[0]
    cube_size = 1.0
    layer_spacing = 120.0  
    cube_height = 40.0 

    colors = ['#ffcccc', '#cce5ff', '#ccffcc', '#ffe6cc', '#e6e6fa']

    for z in range(n):
        for y in range(n):
            for x in range(n):
                x_pos, y_pos, z_pos = x, y, z * layer_spacing + cube_height
                face_color = colors[z % len(colors)]

                ax.bar3d(x_pos, y_pos, z_pos, cube_size, cube_size, cube_height, color=face_color, alpha=0.15)
                
                ax.text(x_pos + cube_size / 2, y_pos + cube_size / 2, z_pos + cube_height/2, 
                        str(cube[z, y, x]), ha='center', va='center', color="black", fontsize=6, weight='bold', 
                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer (Z)')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_zlim(-0.5, n * layer_spacing + cube_height)

    ax.view_init(elev=20, azim=30)

    plt.title("3D Visualization of The Magic Cube")
    plt.show()


def run_simulation(n, initial_temperature, cooling_rate, stopping_temperature, tolerance):
    initial_cube = MagicCube(n)
    show_cube(initial_cube)
    sa = SimulatedAnnealing(initial_cube, initial_temperature, cooling_rate, stopping_temperature, tolerance)
    best_solution, best_cost, duration = sa.anneal()
    return best_solution, best_cost, duration

n = 5
initial_temperature = 1
cooling_rate = 0.90
stopping_temperature = 1e-100
tolerance = 0

run_simulation(n, initial_temperature, cooling_rate, stopping_temperature, tolerance)